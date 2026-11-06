"""Bet-slip analyser: photo -> OpenCV -> Tesseract -> picks -> model evaluation.

Pipeline
  1. preprocess()  OpenCV: grayscale, upscale, dark-mode inversion, deskew, adaptive threshold
  2. ocr_lines()   Tesseract (spa+eng) -> text lines with confidence
  3. parse_slip()  lines -> picks: match (fuzzy team names), market, selection, line, odds
  4. evaluate()    each pick -> model probability, implied probability, expected value

OCR is never fully reliable on bet slips (every bookmaker has its own layout), so the
parser only *proposes* picks; the dashboard shows them in an editable table to confirm.

How to read the evaluation honestly: the 1X2 backtest shows the betting market is more
accurate than this model (log loss 0.964 vs 0.985). When the model "sees value" in an
odd, the model being wrong is the likeliest explanation. The most trustworthy signal is
the slip's odds compared with the market consensus for the same match.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import pytesseract
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from futbol.config import get_settings
from futbol.markets import MatchForecaster, p_total_over, player_over_probability

# --------------------------------------------------------------------------- 1. OpenCV

TARGET_WIDTH = 1600  # Tesseract reads best with ~25-35 px tall characters


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen (¿formato no soportado?)")
    return img


def deskew_angle(binary_inv: np.ndarray) -> float:
    """Angle of the text block from the minimum-area rectangle around ink pixels."""
    coords = np.column_stack(np.where(binary_inv > 0)).astype(np.float32)
    if len(coords) < 500:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV returns angles in (0, 90]; map to the small correction we actually want.
    if angle > 45:
        angle -= 90
    return float(angle) if abs(angle) < 15 else 0.0


def _deskew(images: list[np.ndarray], reference: np.ndarray) -> list[np.ndarray]:
    angle = deskew_angle(cv2.bitwise_not(reference))
    if abs(angle) <= 0.3:
        return images
    h, w = reference.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return [cv2.warpAffine(im, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255) for im in images]


def preprocess_variants(img: np.ndarray) -> dict[str, np.ndarray]:
    """BGR photo/screenshot -> three OCR-ready images that fail in different ways.

    * gray:       upscaled, inverted when dark; keeps faint grey text (odds columns)
    * clahe_otsu: local contrast boost + global threshold; crisp on screenshots
    * adaptive:   local threshold; survives shadows and uneven light in phone photos
    Reading all three and voting beats any single one (see parse_slip_variants).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = TARGET_WIDTH / gray.shape[1]
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if gray.mean() < 110:  # dark-mode screenshot: light text on dark background
        gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    clahe_otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=15,
    )  # fmt: skip
    g, c, a = _deskew([gray, clahe_otsu, adaptive], adaptive)
    return {"gray": g, "clahe_otsu": c, "adaptive": a}


def preprocess(img: np.ndarray) -> np.ndarray:
    """Single cleaned image (the one shown in the dashboard)."""
    return preprocess_variants(img)["clahe_otsu"]


# --------------------------------------------------------------------------- 2. Tesseract


@dataclass
class OcrLine:
    text: str
    confidence: float


def ocr_lines(binary: np.ndarray) -> list[OcrLine]:
    settings = get_settings()
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    # Point Tesseract at the project's language models through the environment: pytesseract
    # passes a quoted --tessdata-dir literally on Windows, quotes included.
    os.environ["TESSDATA_PREFIX"] = settings.tessdata_dir
    data = pytesseract.image_to_data(
        binary,
        lang="spa+eng",
        # psm 6: one uniform block. psm 4 (single column) drops right-aligned odds columns
        # into separate blocks and often loses them; psm 6 keeps them on their selection's line.
        config="--psm 6",
        output_type=pytesseract.Output.DATAFRAME,
    )
    data = data[(data.conf >= 0) & data.text.notna()]
    data = data[data.text.astype(str).str.strip() != ""]
    lines = []
    for _, g in data.groupby(["block_num", "par_num", "line_num"], sort=True):
        text = " ".join(g.text.astype(str))
        lines.append(OcrLine(text=text, confidence=float(g.conf.mean())))
    return lines


# --------------------------------------------------------------------------- 3. parsing

# Common Spanish/long names not covered by the stored aliases (football-data -> extra names).
MANUAL_ALIASES = {
    "Man United": ["Manchester United", "Man Utd", "Manchester Utd"],
    "Man City": ["Manchester City"],
    "Nott'm Forest": ["Nottingham Forest", "Nottingham"],
    "Newcastle": ["Newcastle United"],
    "Tottenham": ["Tottenham Hotspur", "Spurs"],
    "Brighton": ["Brighton & Hove Albion", "Brighton and Hove Albion"],
    "Leeds": ["Leeds United"],
    "Ipswich": ["Ipswich Town"],
    "Hull": ["Hull City"],
    "Coventry": ["Coventry City"],
    "Ath Madrid": ["Atletico Madrid", "Atletico de Madrid", "Atleti"],
    "Ath Bilbao": ["Athletic Bilbao", "Athletic Club"],
    "Betis": ["Real Betis"],
    "Sociedad": ["Real Sociedad"],
    "Vallecano": ["Rayo Vallecano", "Rayo"],
    "Espanol": ["Espanyol"],
    "Celta": ["Celta de Vigo", "Celta Vigo"],
    "La Coruna": ["Deportivo La Coruna", "Deportivo"],
    "Santander": ["Racing de Santander", "Racing Santander"],
    "Bayern Munich": ["Bayern Munchen", "Bayern Munchen", "Bayern Munich", "Bayern"],
    "Dortmund": ["Borussia Dortmund", "BVB"],
    "M'gladbach": ["Monchengladbach", "Borussia Monchengladbach", "Gladbach"],
    "Leverkusen": ["Bayer Leverkusen"],
    "RB Leipzig": ["Leipzig"],
    "Ein Frankfurt": ["Eintracht Frankfurt"],
    "FC Koln": ["Koln", "Colonia", "Cologne"],
    "Inter": ["Inter de Milan", "Inter Milan", "Internazionale"],
    "Milan": ["AC Milan"],
    "Napoli": ["Napoles", "SSC Napoli"],
    "Roma": ["AS Roma"],
    "Juventus": ["Juve"],
    "Paris SG": ["PSG", "Paris Saint-Germain", "Paris Saint Germain"],
    "Marseille": ["Olympique de Marsella", "Marsella", "Olympique Marseille"],
    "Lyon": ["Olympique de Lyon", "Olympique Lyonnais"],
    "Monaco": ["AS Monaco"],
    # Liga MX
    "Club America": ["America", "Club América", "Águilas del América"],
    "Guadalajara Chivas": ["Chivas", "Guadalajara", "Chivas Guadalajara", "CD Guadalajara"],
    "UNAM Pumas": ["Pumas", "Pumas UNAM", "U.N.A.M."],
    "Tigres UANL": ["Tigres", "UANL"],
    "Monterrey": ["Rayados", "CF Monterrey"],
    "Club Leon": ["Leon", "León", "Club León"],
    "Club Tijuana": ["Tijuana", "Xolos", "Xolos de Tijuana"],
    "Atl. San Luis": ["Atletico San Luis", "Atlético de San Luis", "San Luis"],
    "Santos Laguna": ["Santos", "Santos Lag."],
    "Queretaro": ["Querétaro", "Gallos Blancos"],
    "Mazatlan FC": ["Mazatlan", "Mazatlán"],
    "Juarez": ["FC Juarez", "FC Juárez", "Bravos de Juárez"],
    "Cruz Azul": ["La Maquina"],
    # Argentina
    "Atl. Tucuman": ["Atletico Tucuman", "Atlético Tucumán"],
    "Estudiantes L.P.": ["Estudiantes", "Estudiantes de La Plata"],
    "Gimnasia L.P.": ["Gimnasia", "Gimnasia La Plata", "Gimnasia y Esgrima La Plata"],
    "Newells Old Boys": ["Newell's", "Newell's Old Boys", "Newells"],
    "Talleres Cordoba": ["Talleres", "Talleres de Córdoba"],
    "Velez Sarsfield": ["Velez", "Vélez", "Vélez Sarsfield"],
    "Argentinos Jrs": ["Argentinos Juniors", "Argentinos"],
    "Ind. Rivadavia": ["Independiente Rivadavia"],
    "Dep. Riestra": ["Deportivo Riestra", "Riestra"],
    "Union de Santa Fe": ["Union", "Unión", "Unión de Santa Fe"],
    "Central Cordoba": ["Central Córdoba", "Central Cordoba SdE"],
    "Huracan": ["Huracán"],
    "Lanus": ["Lanús"],
    # Brasil
    "Athletico-PR": ["Athletico Paranaense", "Athletico", "Athletico PR"],
    "Atletico-MG": ["Atletico Mineiro", "Atlético Mineiro", "Atletico MG"],
    "Botafogo RJ": ["Botafogo"],
    "Flamengo RJ": ["Flamengo"],
    "Bragantino": ["Red Bull Bragantino", "RB Bragantino"],
    "Sao Paulo": ["São Paulo"],
    "Gremio": ["Grêmio"],
    "Vasco": ["Vasco da Gama"],
    "Vitoria": ["Vitória"],
    "Chapecoense-SC": ["Chapecoense"],
}

SEPARATORS = re.compile(r"\s+(?:vs\.?|v\.?|v/s|-|–|—|x|@|contra)\s+", re.IGNORECASE)
DECIMAL_ODDS = re.compile(r"(?<![\d.,/$])(\d{1,3}[.,]\d{2})(?![\d.,])")  # not "$100.00"
AMERICAN_ODDS = re.compile(r"(?<![\w.,/])([+-]\d{3,4})(?![\d.,])")
FRACTIONAL_ODDS = re.compile(r"(?<![\d.,/])(\d{1,2})/(\d{1,2})(?![\d/])")
LINE = re.compile(r"(\d{1,2}[.,]5)")


def normalize(text: str) -> str:
    """Lowercase, no accents, no punctuation: what fuzzy matching compares."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TeamIndex:
    """Every known name of every team -> (football-data name, league)."""

    choices: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Names that point to several teams ("Santos": Brazil's Santos and Liga MX's Santos Laguna).
    alternatives: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def _add(self, name: str, team: tuple[str, str], primary: bool = False) -> None:
        key = normalize(name)
        if primary or key not in self.choices:
            if key in self.choices and self.choices[key] != team:
                self.alternatives.setdefault(key, [self.choices[key]]).append(team)
            self.choices[key] = team
        elif self.choices[key] != team:
            alts = self.alternatives.setdefault(key, [self.choices[key]])
            if team not in alts:
                alts.append(team)

    @classmethod
    def build(cls, teams: pd.DataFrame, aliases: pd.DataFrame) -> TeamIndex:
        """teams: league_code, team. aliases: league_code, alias, team."""
        idx = cls()
        league_of = dict(zip(teams.team, teams.league_code, strict=True))
        for t, lg in league_of.items():  # official names first: they win a clash
            idx._add(t, (t, lg), primary=True)
        for t, lg in league_of.items():
            for extra in MANUAL_ALIASES.get(t, []):
                idx._add(extra, (t, lg))
        for a in aliases.itertuples(index=False):
            if a.team in league_of:
                idx._add(a.alias, (a.team, league_of[a.team]))
        return idx

    def readings(self, text: str) -> list[tuple[str, str]]:
        """Every team an exact-matching name could mean (one, unless the name is ambiguous)."""
        hit = self.exact(text)
        if hit is None:
            return []
        hit_key = next(
            k for k, v in self.choices.items() if v == hit and fuzz.ratio(k, normalize(text)) >= 88
        )
        return self.alternatives.get(hit_key, [hit])

    def match(self, text: str, cutoff: float = 85) -> tuple[str, str, float] | None:
        """Best team for a fragment of text, or None below the similarity cutoff."""
        q = normalize(DECIMAL_ODDS.sub(" ", text))
        if len(q) < 3:
            return None
        hit = process.extractOne(q, list(self.choices), scorer=fuzz.WRatio, score_cutoff=cutoff)
        if not hit:
            return None
        team, league = self.choices[hit[0]]
        return team, league, float(hit[1])

    def exact(self, text: str, cutoff: float = 88) -> tuple[str, str] | None:
        """Whole-string match (no partial credit): for lines that should be just a team name."""
        q = normalize(text)
        if len(q) < 3:
            return None
        hit = process.extractOne(q, list(self.choices), scorer=fuzz.ratio, score_cutoff=cutoff)
        return self.choices[hit[0]] if hit else None

    def by_code(self, code: str) -> list[tuple[str, str]]:
        """Teams a 2-4 letter abbreviation can stand for: VAL -> Valencia, RMA -> Real Madrid."""
        code = code.lower()
        found = []
        for name, team in self.choices.items():
            words = [w for w in name.replace("'", "").split() if w]
            if not words:
                continue
            codes = {"".join(words)[:3], words[-1][:3], "".join(w[0] for w in words)}
            if len(words) >= 2:
                codes.add(words[0][0] + words[1][:2])
            if code in codes and team not in found:
                found.append(team)
        return found

    def mentioned(self, text: str, candidates: list[str], cutoff: float = 85) -> str | None:
        """Which of `candidates` (usually the match's two teams) the text names, if any."""
        q = normalize(text)
        best, best_score = None, cutoff
        for cand in candidates:
            names = [n for n, (t, _) in self.choices.items() if t == cand]
            for n in names:
                s = (
                    fuzz.partial_ratio(n, q)
                    if len(n) >= 4
                    else (100 if f" {n} " in f" {q} " else 0)
                )
                if s > best_score:
                    best, best_score = cand, s
        return best


def parse_odds(text: str) -> float | None:
    """Last odds-looking token: decimal (1.85 / 1,85), American (+150 / -120) or 5/2."""
    found: list[tuple[int, float]] = []
    for m in DECIMAL_ODDS.finditer(text):
        v = float(m.group(1).replace(",", "."))
        if 1.01 <= v <= 500:
            found.append((m.start(), v))
    for m in AMERICAN_ODDS.finditer(text):
        v = int(m.group(1))
        if abs(v) >= 100:
            found.append((m.start(), 1 + v / 100 if v > 0 else 1 + 100 / abs(v)))
    for m in FRACTIONAL_ODDS.finditer(text):
        num, den = int(m.group(1)), int(m.group(2))
        if den:
            found.append((m.start(), 1 + num / den))
    return round(max(found)[1], 3) if found else None


def has_keyword(q: str, keywords: tuple[str, ...]) -> bool:
    """Keyword test that survives OCR eating the spaces ('TIROS ALARCO' still says 'al arco')."""
    squeezed = q.replace(" ", "")
    return any(k in q or (" " in k and k.replace(" ", "") in squeezed) for k in keywords)


MARKET_KEYWORDS = [
    ("sot", ("a puerta", "al arco", "on target", "a porteria")),
    ("corners", ("corner", "esquina")),
    ("yellow", ("tarjeta", "amarilla", "card", "booking")),
    ("shots", ("tiros", "remates", "shots")),
]


def parse_selection(
    text: str, home: str, away: str, teams: TeamIndex
) -> tuple[str, str, float | None]:
    """(market, selection, line) from the text around a match. Unknown -> ('?', '?', None)."""
    q = normalize(text)
    line_match = LINE.search(text.replace(" ", ""))
    line = float(line_match.group(1).replace(",", ".")) if line_match else None

    if ("ambos" in q and ("marcan" in q or "anotan" in q)) or "both teams" in q or "btts" in q:
        return "btts", "no" if re.search(r"\bno\b", q) else "yes", None

    over = re.search(r"\b(mas|over|alta|encima)\b", q) or "+" in text.replace("+1", "")
    under = re.search(r"\b(menos|under|baja|debajo)\b", q)
    if line is not None and (over or under):
        market = next((m for m, kws in MARKET_KEYWORDS if has_keyword(q, kws)), "goals")
        return market, "under" if under else "over", line

    if "doble oportunidad" in q or "double chance" in q or re.search(r"\b(1x|x2|12)\b", q):
        for tag in ("1x", "x2", "12"):
            if re.search(rf"\b{tag}\b", q):
                return "double", tag.upper(), None
        named = teams.mentioned(text, [home, away])
        if named:
            return "double", "1X" if named == home else "X2", None

    if re.search(r"\b(empate|draw)\b", q) or re.search(r"(?:^|\s)x(?:\s|$)", q):
        return "1x2", "draw", None
    named = teams.mentioned(text, [home, away])
    if named:
        return "1x2", "home" if named == home else "away", None
    return "?", "?", None


def find_match(line: str, teams: TeamIndex) -> tuple[str, str, str | None] | None:
    """'Arsenal vs Hull' -> (home, away, league). League None when the teams' leagues differ."""
    parts = SEPARATORS.split(line, maxsplit=1)
    if len(parts) != 2:
        return None
    a, b = teams.match(parts[0]), teams.match(parts[1])
    if not a or not b or a[0] == b[0]:
        return None
    return a[0], b[0], a[1] if a[1] == b[1] else None


PICK_COLUMNS = [
    "bet_type", "home_team", "away_team", "league_code", "market", "selection", "line", "odds",
    "player", "player_team", "player_avg", "text",
]  # fmt: skip

# Player props: market key -> (team-level model used for the opponent adjustment, keywords).
PLAYER_MARKETS = {
    "player_sot": ("sot", ("a puerta", "al arco", "on target", "a porteria")),
    "player_shots": ("shots", ("tiros", "remates", "disparos", "shots")),
    "player_goals": (
        "goals",
        ("anotara", "anota", "marcara", "marca", "goleador", "anytime", "anotar", "marcar"),
    ),
    "player_cards": ("yellow", ("tarjeta", "amonestado", "booked", "amarilla")),
}
# Lines that are never a player's name.
NOT_A_NAME = re.compile(
    r"resultado|total|ambos|doble|oportunidad|goles|corner|tarjetas|boleto|combinada|cuota|handicap|"
    r"mas de|menos de|tiros|a puerta|^si$|^no$",
    re.IGNORECASE,
)


BTTS = re.compile(r"ambos|both teams|btts")
CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")
PLAYER_CUT = re.compile(
    r"[\d+@]|\b(?:tiros?|remates?|disparos?|a puerta|anotar[aá]?|marcar[aá]?|goleador|tarjeta|amonestado|"
    r"m[aá]s de|menos de|over|under)\b",
    re.IGNORECASE,
)


def name_and_code(text: str) -> tuple[str | None, str | None]:
    """'awa K. Mbappé RMA |' -> ('K. Mbappé', 'RMA').

    Keeps capitalised words and initials; drops OCR junk from the player's photo (lowercase
    fragments, short all-caps bits before the name) and returns the team code after it.
    """
    name_tokens, code = [], None
    for token in text.replace("|", " ").split():
        bare = token.strip("()[]{}'\"‘’“”,;:")
        if re.fullmatch(r"[A-Z]{2,4}", bare):
            if name_tokens:
                code = bare
                break
            continue
        initial = re.fullmatch(r"[A-ZÁÉÍÓÚÑ]\.\S*", bare)
        capitalised = bare[:1].isupper() and any(ch.islower() for ch in bare)
        if initial or capitalised:
            name_tokens.append(bare)
        elif name_tokens:
            break
    return (" ".join(name_tokens) or None), code


def parse_player_prop(lines: list[str], teams: TeamIndex) -> tuple[str, float, str] | None:
    """(market, line, player name) when the text is a player prop, e.g.
    'Kylian Mbappé' + 'Tiros a puerta: 3 o más'  -> ('player_sot', 2.5, 'Kylian Mbappé').
    'N+' / 'N o más' means at least N, i.e. over N - 0.5. Scorer / card props default to 0.5.
    """
    q = normalize(" ".join(lines))
    if BTTS.search(q):
        return None  # "Ambos equipos anotan" is a match market, not "a player will score"
    market = next((m for m, (_, kws) in PLAYER_MARKETS.items() if has_keyword(q, kws)), None)
    if market is None:
        return None
    name = None
    code = None
    for line in lines:
        if CLOCK.search(line):
            continue  # "sáb 7 nov · 14:00": a kick-off line, often the neighbouring pick's
        # A name ends where the numbers or the stat start: "Mbappé 3+ tiros a puerta @2.40".
        text = PLAYER_CUT.split(line, maxsplit=1)[0].strip(" -:·|")
        plain = normalize(text)
        if not plain or re.search(r"\d", plain) or NOT_A_NAME.search(plain) or teams.exact(text):
            continue
        name, code = name_and_code(text)
        if name:
            break
    if name is None:
        return None  # e.g. "Total de tiros a puerta": a team market, not a player
    # "3+" (raw text: normalize drops '+') or "3 o más". OCR often reads that "o" as a zero
    # ("3 0 más", "30 más"); the lazy \d{1,2}? still keeps a real "10 o más" as 10.
    # Odds are removed first, or "2.40 Más de 1.5" would read as "4 0 más".
    raw = " ".join(AMERICAN_ODDS.sub(" ", DECIMAL_ODDS.sub(" ", line)) for line in lines)
    at_least = re.search(r"(\d{1,2})(?:[.,]0)?\s*\+", raw) or re.search(  # "3+", "3.0+"
        r"(\d{1,2}?)\s*[o0]\s*mas|(\d{1,2})\s*(?:or more|y mas)", normalize(raw)
    )
    half = LINE.search(raw.replace(" ", ""))
    if at_least:
        line_value = int(next(g for g in at_least.groups() if g)) - 0.5
    elif half:
        line_value = float(half.group(1).replace(",", "."))
    elif market in ("player_goals", "player_cards"):
        line_value = 0.5
    else:
        bare = re.search(r"\b(\d{1,2})\b", q)
        line_value = int(bare.group(1)) - 0.5 if bare else None
    parse_player_prop.last_code = code  # the team code next to the name, if any ("RMA")
    return market, line_value, name


# Slip totals ("Cuota total 16.37", "Apuesta $100", "Ganancia potencial") look like odds:
# they must never be attached to a pick.
FOOTER = re.compile(
    r"cuota total|cuotas? combinad|ganancia|apuesta\b|entrada|boost|monto|importe|pago|premio|stake|"
    r"potential|payout",
    re.IGNORECASE,
)


# Words that mean a line is a selection, not a bare team name ("GANA PA 90' Sevilla").
SELECTION_WORDS = re.compile(
    r"\b(gana|ganador|empate|mas|menos|ambos|tiros|total|doble|resultado|handicap|over|under|al arco)\b"
)
JUNK_TOKEN = re.compile(
    r"^[^A-Za-zÀ-ÿ]*[A-Za-zÀ-ÿ]{0,3}[^A-Za-zÀ-ÿ]*$"
)  # crest debris: "es)", "RS", "©"
CODE_MATCH = re.compile(r"\b([A-Z]{2,4})\s+(?:vs\.?|v|-|–)\s+([A-Z]{2,4})\b")


def team_line(line: str, teams: TeamIndex) -> tuple[str, str] | None:
    """A line that is just a team name, possibly after crest debris ('RS Crystal Palace')."""
    if DECIMAL_ODDS.search(line) or CLOCK.search(line) or SELECTION_WORDS.search(normalize(line)):
        return None
    tokens = line.split()
    for drop in range(3):
        if drop and not JUNK_TOKEN.match(tokens[drop - 1]):
            break
        rest = tokens[drop:]
        if not rest:
            break
        if hit := teams.exact(" ".join(rest)):
            return hit
    return None


def find_match_codes(line: str, teams: TeamIndex, fixtures: pd.DataFrame | None) -> tuple | None:
    """'Mañana (14:00) · VAL vs RMA' -> (home, away, league, {code: team}), using the
    schedule to pick the one real fixture when an abbreviation fits several teams."""
    m = CODE_MATCH.search(line)
    if not m:
        return None
    ca, cb = m.groups()
    pairs = [
        (a, b)
        for a in teams.by_code(ca)
        for b in teams.by_code(cb)
        if a[0] != b[0] and a[1] == b[1]
    ]
    if fixtures is not None and len(pairs) > 1:
        # Both VAL-RMA readings (Valencia, Vallecano) are real fixtures at some point of the
        # season; a slip is about the soonest one, so rank the scheduled pairs by kick-off.
        if "kickoff_utc" in fixtures:
            soonest = fixtures.groupby(["home_team", "away_team"])["kickoff_utc"].min().to_dict()
        else:
            soonest = dict.fromkeys(zip(fixtures.home_team, fixtures.away_team, strict=True), 0)
        when = {(a, b): soonest[(a[0], b[0])] for a, b in pairs if (a[0], b[0]) in soonest}
        pairs = sorted(when, key=when.get)[:1]
    if len(pairs) != 1:
        return None
    (home, league), (away, _) = pairs[0]
    return home, away, league, {ca: home, cb: away}


def _match_in_block(block: list[str], teams: TeamIndex, fixtures) -> tuple | None:
    """(home, away, league, codes, used line positions) for one pick's block of lines."""
    for j, line in enumerate(block):
        if m := find_match(line, teams):
            return (*m, {}, {j})
        if m := find_match_codes(line, teams, fixtures):
            return (*m, {j})
    found: list[tuple[int, tuple[str, str], str]] = []
    for j, line in enumerate(block):
        if (t := team_line(line, teams)) and all(t[0] != f[1][0] for f in found):
            found.append((j, t, line))
        if len(found) == 2:
            (i1, (home, lg1), l1), (i2, (away, lg2), l2) = found
            if lg1 != lg2:  # an ambiguous name? prefer the reading where both share a league
                for h in _line_readings(l1, teams) or [(home, lg1)]:
                    for a in _line_readings(l2, teams) or [(away, lg2)]:
                        if h[1] == a[1] and h[0] != a[0]:
                            return h[0], a[0], h[1], {}, {i1, i2}
            return home, away, lg1 if lg1 == lg2 else None, {}, {i1, i2}
    return None


def _line_readings(line: str, teams: TeamIndex) -> list[tuple[str, str]]:
    tokens = line.split()
    for drop in range(3):
        if drop and not JUNK_TOKEN.match(tokens[drop - 1]):
            break
        if tokens[drop:] and (readings := teams.readings(" ".join(tokens[drop:]))):
            return readings
    return []


def _pick_row(block: list[str], match: tuple, teams: TeamIndex) -> dict:
    home, away, league, codes, used = match
    details = [ln for j, ln in enumerate(block) if j not in used and not FOOTER.search(ln)]
    context = " ".join(details)
    parse_player_prop.last_code = None
    prop = parse_player_prop(details, teams)
    player_team = None
    if prop:
        market, line_value, player = prop
        selection, bet_type = "over", "Jugador"
        player_team = codes.get(parse_player_prop.last_code)
    else:
        market, selection, line_value = parse_selection(context, home, away, teams)
        player, bet_type = None, "Partido"
    odds = next((o for ln in details if (o := parse_odds(ln))), None)
    if odds is None:
        odds = next((o for j, ln in enumerate(block) if j in used and (o := parse_odds(ln))), None)
    return {
        "bet_type": bet_type, "home_team": home, "away_team": away, "league_code": league,
        "market": market, "selection": selection, "line": line_value, "odds": odds,
        "player": player, "player_team": player_team, "player_avg": None, "text": " | ".join(block),
    }  # fmt: skip


def _blocks_by_match_lines(lines: list[str], teams: TeamIndex, fixtures) -> list[list[list[str]]]:
    """Layouts with a 'Home vs Away' line per pick: details either above or below it."""
    keys = [
        i
        for i, ln in enumerate(lines)
        if find_match(ln, teams) or find_match_codes(ln, teams, fixtures)
    ]
    if not keys:
        return []
    below = [lines[k:nxt] for k, nxt in zip(keys, [*keys[1:], len(lines)], strict=True)]
    above = [lines[prev + 1 : k + 1] for k, prev in zip(keys, [-1, *keys[:-1]], strict=True)]
    return [below, above]


def _blocks_by_kickoff(lines: list[str]) -> list[list[list[str]]]:
    """Layouts where every pick carries a kick-off time ('08/nov (11:30)'), which is a far
    steadier separator than team lines: it opens each block on some apps, closes it on others."""
    kicks = [i for i, ln in enumerate(lines) if CLOCK.search(ln)]
    if len(kicks) < 2:
        return []
    starts = [lines[k:nxt] for k, nxt in zip(kicks, [*kicks[1:], len(lines)], strict=True)]
    ends = [lines[prev + 1 : k + 1] for k, prev in zip(kicks, [-1, *kicks[:-1]], strict=True)]
    return [starts, ends]


def parse_slip(
    lines: list[str], teams: TeamIndex, fixtures: pd.DataFrame | None = None
) -> pd.DataFrame:
    """OCR lines -> one row per detected pick.

    Several ways of cutting the slip into one block per pick are tried (by kick-off line,
    by match line, details above or below) and the reading with the most recognised picks,
    selections and odds wins - the parser adapts to the bookmaker's layout.
    """
    best, best_score = pd.DataFrame(columns=PICK_COLUMNS), -1
    for blocks in [*_blocks_by_kickoff(lines), *_blocks_by_match_lines(lines, teams, fixtures)]:
        rows = []
        for block in blocks:
            if match := _match_in_block(block, teams, fixtures):
                rows.append(_pick_row(block, match, teams))
        df = pd.DataFrame(rows, columns=PICK_COLUMNS)
        score = (
            3 * len(df) + int((df.market != "?").sum() + df.odds.notna().sum()) if len(df) else 0
        )
        if score > best_score:
            best, best_score = df, score
    return best


def parse_slip_variants(
    variants: list[list[str]], teams: TeamIndex, fixtures: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Parse each OCR variant, then vote: per match the most supported pick, and per pick
    the most frequent well-formed odds. One variant misreading '1.67x' as '167x' or
    '2.11x' as '2.5x' is outvoted by the others."""
    frames = [parse_slip(v, teams, fixtures).assign(_variant=i, _pos=lambda d: range(len(d)))
              for i, v in enumerate(variants)]  # fmt: skip
    allp = pd.concat(frames, ignore_index=True)
    if allp.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    allp["_key"] = (
        allp.home_team + "|" + allp.away_team + "|" + allp.market.astype(str) + "|"
        + allp.selection.astype(str) + "|" + allp.player.fillna("")
    )  # fmt: skip
    rows = []
    for (_, _), match_rows in allp.groupby(["home_team", "away_team"], sort=False):
        support = match_rows.groupby("_key")["_variant"].nunique()
        chosen = match_rows[match_rows._key == support.idxmax()]
        row = chosen.iloc[0].to_dict()
        odds = chosen.odds.dropna().round(2)
        if not odds.empty:
            row["odds"] = (
                float(odds.mode().iloc[0])
                if odds.value_counts().iloc[0] > 1
                else float(odds.iloc[0])
            )
        lines = chosen.line.dropna()
        if not lines.empty:
            row["line"] = float(lines.mode().iloc[0])
        row["player_team"] = (
            chosen.player_team.dropna().iloc[0] if chosen.player_team.notna().any() else None
        )
        row["_order"] = match_rows._pos.min()
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("_order")
    return out[PICK_COLUMNS].reset_index(drop=True)


@dataclass
class SlipMeta:
    stake: float | None = None  # "Entrada $50.00"
    boost_pct: float | None = None  # "Boost +80%"
    printed_total: float | None = None  # the combined odds as printed (boosted, if a boost applies)

    @property
    def base_total(self) -> float | None:
        """Combined odds before the boost: what the product of the picks' odds should equal."""
        if self.printed_total is None:
            return None
        return self.printed_total / (1 + (self.boost_pct or 0) / 100)


def _money(text: str) -> float:
    return float(text.replace(",", ""))


def parse_meta(lines: list[str]) -> SlipMeta:
    meta = SlipMeta()
    joined = " ".join(lines)
    if m := re.search(
        r"(?:entrada|apuesta|importe|stake)\D{0,6}\$?\s*(\d[\d,]*(?:\.\d{2})?)", joined, re.I
    ):
        meta.stake = _money(m.group(1))
    if m := re.search(r"boost\s*\+?\s*(\d{1,3})\s*%", joined, re.I):
        meta.boost_pct = float(m.group(1))
    if m := re.search(r"boost[^:]{0,20}:\s*(\d[\d,]*\.\d{2})\s*[xX]", joined, re.I):
        meta.printed_total = _money(m.group(1))
    else:
        meta.printed_total = parse_total_odds(lines)
    return meta


TOTAL_ODDS_LINE = re.compile(r"cuota total|cuotas? combinad|total odds", re.IGNORECASE)


def parse_total_odds(lines: list[str]) -> float | None:
    """The slip's printed combined odds, if any ("Cuota total 18.08").

    OCR often splits numbers ("1 8.08"), so spaces between digits are removed first.
    """
    for line in lines:
        if TOTAL_ODDS_LINE.search(line):
            compact = re.sub(r"(?<=\d)\s+(?=[\d.,])", "", line)
            if (odds := parse_odds(compact)) is not None:
                return odds
    return None


def odds_consistency(
    picks: pd.DataFrame, printed_total: float | None, tolerance: float = 0.02
) -> str | None:
    """Warn when the product of the read odds doesn't match the slip's printed total.

    A single misread digit (1.90 read as 4.90) breaks the product, so this catches most
    OCR mistakes in the odds without knowing which one is wrong.
    """
    if printed_total is None or picks.empty or picks.odds.isna().any():
        return None
    product = float(picks.odds.prod())
    if abs(product / printed_total - 1) <= tolerance:
        return None
    return (
        f"Las cuotas leídas multiplicadas dan {product:.2f}, pero el boleto dice cuota total "
        f"{printed_total:.2f}. Probablemente el OCR leyó mal alguna cuota: revísalas."
    )


def suspect_odds(picks: pd.DataFrame, printed_total: float) -> tuple[int, float] | None:
    """Which single pick's odds, if corrected, would make the product match the printed total.

    For each pick, the odds it *should* have = total / product of the others. The suspect is
    the pick whose read odds differ from that value by one character (a misread digit).
    Returns (row position, corrected odds) or None when no single fix explains it.
    """
    if printed_total is None or picks.empty or picks.odds.isna().any():
        return None
    odds = picks.odds.to_numpy(float)
    best = None
    for pos, read in enumerate(odds):
        needed = printed_total / (odds.prod() / read)
        if not 1.01 <= needed <= 100:
            continue
        distance = Levenshtein.distance(f"{read:.2f}", f"{needed:.2f}")
        if distance <= 1 and (best is None or distance < best[2]):
            best = (pos, round(needed, 2), distance)
    return (best[0], best[1]) if best else None


# --------------------------------------------------------------------------- 4. evaluation

MARKET_NAMES = {
    "1x2": "Resultado (1X2)",
    "double": "Doble oportunidad",
    "goals": "Goles más/menos",
    "btts": "Ambos marcan",
    "corners": "Córners más/menos",
    "yellow": "Amarillas más/menos",
    "shots": "Tiros más/menos",
    "sot": "Tiros a puerta más/menos",
    "player_sot": "Jugador: tiros a puerta",
    "player_shots": "Jugador: tiros",
    "player_goals": "Jugador: goles",
    "player_cards": "Jugador: tarjeta amarilla",
}
SELECTION_NAMES = {
    "home": "Gana local", "draw": "Empate", "away": "Gana visitante",
    "1X": "Local o empate", "X2": "Empate o visitante", "12": "Local o visitante",
    "over": "Más de", "under": "Menos de", "yes": "Sí", "no": "No",
}  # fmt: skip
VALID_SELECTIONS = {
    "1x2": {"home", "draw", "away"},
    "double": {"1X", "X2", "12"},
    "goals": {"over", "under"},
    "btts": {"yes", "no"},
    "corners": {"over", "under"},
    "yellow": {"over", "under"},
    "shots": {"over", "under"},
    "sot": {"over", "under"},
    **{m: {"over", "under"} for m in PLAYER_MARKETS},
}
COUNT_MARKETS = {"goals", "corners", "yellow", "shots", "sot", *PLAYER_MARKETS}


def pick_probability(
    fc: MatchForecaster, home: str, away: str, market: str, selection: str, line: float | None
) -> float:
    """Model probability that the pick wins."""
    fixture = pd.DataFrame({"home_team": [home], "away_team": [away]})
    if market in ("1x2", "double", "goals", "btts"):
        m = fc.score_matrix(home, away)
        i, j = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
        p_home, p_draw = float(m[i > j].sum()), float(np.trace(m))
        p_away = 1 - p_home - p_draw
        if market == "1x2":
            return {"home": p_home, "draw": p_draw, "away": p_away}[selection]
        if market == "double":
            return {"1X": p_home + p_draw, "X2": p_draw + p_away, "12": p_home + p_away}[selection]
        if market == "btts":
            p = float(m[(i > 0) & (j > 0)].sum())
            return p if selection == "yes" else 1 - p
        p_over = float(m[(i + j) > line].sum())
        return p_over if selection == "over" else 1 - p_over
    c = fc.counts(market, fixture, line).iloc[0]
    p_over = float(p_total_over(np.array([c.exp_home]), np.array([c.exp_away]), line)[0])
    return p_over if selection == "over" else 1 - p_over


def check_pick(pick: pd.Series) -> str | None:
    """Why a pick can't be evaluated, or None if it can."""
    if (
        not pick.home_team
        or not pick.away_team
        or pd.isna(pick.home_team)
        or pd.isna(pick.away_team)
    ):
        return "faltan los equipos"
    if pd.isna(pick.league_code) or not pick.league_code:
        return "los equipos no son de la misma liga (o no se reconocieron)"
    if pick.market not in VALID_SELECTIONS or pick.selection not in VALID_SELECTIONS[pick.market]:
        return "mercado o selección no reconocidos"
    if pick.market in COUNT_MARKETS and pd.isna(pick.line):
        return "falta la línea (p. ej. 2.5)"
    if pick.market in PLAYER_MARKETS:
        if pick.get("player_team") not in (pick.home_team, pick.away_team):
            return "elige el equipo del jugador (local o visitante)"
        if pd.isna(pick.get("player_avg")) or pick.get("player_avg") is None:
            return "falta el promedio del jugador por partido (no hay datos públicos de jugadores)"
    if pd.isna(pick.odds) or pick.odds <= 1:
        return "falta la cuota"
    return None


def evaluate(picks: pd.DataFrame, get_forecaster: Callable[[str], MatchForecaster]) -> pd.DataFrame:
    """Add model probability, implied probability, fair odds and expected value per pick."""
    out = picks.copy()
    for col in ["p_model", "p_implied", "fair_odds", "ev", "opponent_factor", "problem"]:
        out[col] = np.nan if col != "problem" else None
    for idx, pick in out.iterrows():
        problem = check_pick(pick)
        if problem:
            out.at[idx, "problem"] = problem
            continue
        fc = get_forecaster(pick.league_code)
        if pick.market in PLAYER_MARKETS:
            p_over, factor = player_over_probability(
                fc, PLAYER_MARKETS[pick.market][0], pick.player_team, pick.home_team,
                pick.away_team, float(pick.player_avg), pick.line,
            )  # fmt: skip
            p = p_over if pick.selection == "over" else 1 - p_over
            out.at[idx, "opponent_factor"] = factor
        else:
            p = pick_probability(
                fc, pick.home_team, pick.away_team, pick.market, pick.selection, pick.line
            )
        out.at[idx, "p_model"] = p
        out.at[idx, "p_implied"] = 1 / pick.odds
        out.at[idx, "fair_odds"] = 1 / p if p > 0 else np.inf
        out.at[idx, "ev"] = p * pick.odds - 1  # expected profit per unit staked
    return out


@dataclass
class SlipSummary:
    n_picks: int
    p_win: float  # model probability that every pick wins
    odds: float  # combined odds of the parlay
    ev: float  # expected profit per unit staked
    same_match_picks: bool  # correlated picks: independence assumption breaks


def summarize(evaluated: pd.DataFrame) -> SlipSummary | None:
    ok = evaluated.dropna(subset=["p_model"])
    if ok.empty:
        return None
    p_win = float(ok.p_model.prod())
    odds = float(ok.odds.prod())
    same = ok.duplicated(["home_team", "away_team"]).any()
    return SlipSummary(len(ok), p_win, odds, p_win * odds - 1, bool(same))


def risk_label(p: float) -> str:
    if p >= 0.6:
        return "Probable"
    if p >= 0.4:
        return "Moneda al aire"
    if p >= 0.2:
        return "Arriesgada"
    if p >= 0.05:
        return "Muy arriesgada"
    return "Lotería"
