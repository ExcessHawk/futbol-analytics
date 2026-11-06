-- Database schema. Idempotent: safe to run on every `scripts/init_db.py`.
--
-- Two source domains live side by side:
--   * StatsBomb (event data): competitions, teams, players, matches, lineups, events
--   * football-data.co.uk (results + bookmaker odds): fd_matches
-- team_xref links them; predictions stores model output; *_stats are derived metrics.

-- ---------------------------------------------------------------- StatsBomb

CREATE TABLE IF NOT EXISTS competitions (
    competition_id   integer NOT NULL,
    season_id        integer NOT NULL,
    competition_name text    NOT NULL,
    season_name      text    NOT NULL,          -- e.g. '2015/2016'
    country_name     text,
    PRIMARY KEY (competition_id, season_id)
);

CREATE TABLE IF NOT EXISTS teams (
    team_id   integer PRIMARY KEY,               -- StatsBomb id
    team_name text NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    player_id   integer PRIMARY KEY,             -- StatsBomb id
    player_name text NOT NULL,
    nickname    text                              -- short name, e.g. 'Riyad Mahrez'
);

CREATE TABLE IF NOT EXISTS matches (
    match_id       integer PRIMARY KEY,
    competition_id integer  NOT NULL,
    season_id      integer  NOT NULL,
    match_date     date     NOT NULL,
    kick_off       time,
    match_week     smallint,
    home_team_id   integer  NOT NULL REFERENCES teams,
    away_team_id   integer  NOT NULL REFERENCES teams,
    home_score     smallint,
    away_score     smallint,
    FOREIGN KEY (competition_id, season_id) REFERENCES competitions
);
CREATE INDEX IF NOT EXISTS ix_matches_comp_date ON matches (competition_id, season_id, match_date);

-- One row per player who actually played in a match.
CREATE TABLE IF NOT EXISTS lineups (
    match_id       integer NOT NULL REFERENCES matches ON DELETE CASCADE,
    player_id      integer NOT NULL REFERENCES players,
    team_id        integer NOT NULL REFERENCES teams,
    jersey_number  smallint,
    is_starter     boolean NOT NULL,
    minutes_played real    NOT NULL,
    main_position  text,                          -- position with most minutes in the match
    PRIMARY KEY (match_id, player_id)
);

-- Typed subset of the StatsBomb event JSON. Coordinates are in StatsBomb units
-- (pitch 120 x 80) and always from the acting team's perspective (attacking towards x=120).
-- 'Ball Receipt*' events are not stored (~26% of rows, no use in this project).
CREATE TABLE IF NOT EXISTS events (
    event_id           uuid PRIMARY KEY,
    match_id           integer  NOT NULL REFERENCES matches ON DELETE CASCADE,
    idx                integer  NOT NULL,         -- order within the match
    period             smallint NOT NULL,
    minute             smallint NOT NULL,
    second             smallint NOT NULL,
    type               text     NOT NULL,         -- Pass, Shot, Carry, Pressure, Duel...
    sub_type           text,                      -- pass/shot/duel/goalkeeper type
    outcome            text,                      -- NULL on a pass means completed
    team_id            integer  NOT NULL,
    player_id          integer,
    position           text,
    possession         integer,
    possession_team_id integer,
    play_pattern       text,
    x                  real,
    y                  real,
    end_x              real,
    end_y              real,
    under_pressure     boolean  NOT NULL DEFAULT false,
    counterpress       boolean  NOT NULL DEFAULT false,
    body_part          text,
    pass_length        real,
    pass_height        text,
    pass_recipient_id  integer,
    pass_cross         boolean  NOT NULL DEFAULT false,
    pass_switch        boolean  NOT NULL DEFAULT false,
    pass_through_ball  boolean  NOT NULL DEFAULT false,
    pass_shot_assist   boolean  NOT NULL DEFAULT false,
    pass_goal_assist   boolean  NOT NULL DEFAULT false,
    shot_xg            real,
    shot_key_pass_id   uuid
);
CREATE INDEX IF NOT EXISTS ix_events_match  ON events (match_id);
CREATE INDEX IF NOT EXISTS ix_events_player ON events (player_id) WHERE player_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_events_shots  ON events (match_id) WHERE type = 'Shot';

-- ------------------------------------------------------- football-data.co.uk

-- Results and odds. Unplayed fixtures have NULL goals and get updated once played.
CREATE TABLE IF NOT EXISTS fd_matches (
    league_code text     NOT NULL,                -- E0, SP1, D1, I1, F1
    season      text     NOT NULL,                -- '2015/2016'
    match_date  date     NOT NULL,
    home_team   text     NOT NULL,
    away_team   text     NOT NULL,
    home_goals  smallint,
    away_goals  smallint,
    home_shots  smallint,
    away_shots  smallint,
    odds_home   real,
    odds_draw   real,
    odds_away   real,
    odds_source text,                             -- which bookmaker column was used
    PRIMARY KEY (league_code, season, home_team, away_team)
);
CREATE INDEX IF NOT EXISTS ix_fd_matches_date ON fd_matches (league_code, match_date);

-- --------------------------------------------------------------- cross-ref

-- football-data team name -> StatsBomb team id, per league.
CREATE TABLE IF NOT EXISTS team_xref (
    league_code text    NOT NULL,
    fd_team     text    NOT NULL,
    team_id     integer NOT NULL REFERENCES teams,
    match_votes integer NOT NULL,                 -- fixtures that agreed on this mapping
    PRIMARY KEY (league_code, fd_team)
);

-- StatsBomb competition -> football-data league code.
CREATE TABLE IF NOT EXISTS competition_xref (
    competition_id integer PRIMARY KEY,
    league_code    text NOT NULL UNIQUE
);
INSERT INTO competition_xref VALUES (2, 'E0'), (11, 'SP1'), (9, 'D1'), (12, 'I1'), (7, 'F1')
ON CONFLICT DO NOTHING;

-- --------------------------------------------------------------- modelling

CREATE TABLE IF NOT EXISTS predictions (
    experiment  text NOT NULL,                    -- multi_season | xg_vs_goals | upcoming
    model       text NOT NULL,
    league_code text NOT NULL,
    season      text NOT NULL,
    match_date  date NOT NULL,
    home_team   text NOT NULL,
    away_team   text NOT NULL,
    p_home      real NOT NULL,
    p_draw      real NOT NULL,
    p_away      real NOT NULL,
    exp_home    real,                             -- expected goals (lambda, mu)
    exp_away    real,
    PRIMARY KEY (experiment, model, league_code, season, home_team, away_team)
);

-- ------------------------------------------------------------- migrations
-- Added after the first version; ADD COLUMN IF NOT EXISTS keeps re-runs idempotent.
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS home_sot   smallint;     -- shots on target
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS away_sot   smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS updated_at timestamptz;  -- last time the row was refreshed

-- Hyperparameters chosen by the last tuning run, so daily updates can skip tuning.
CREATE TABLE IF NOT EXISTS model_params (
    name     text PRIMARY KEY,
    value    double precision NOT NULL,
    tuned_at timestamptz NOT NULL DEFAULT now()
);

-- Match statistics and over/under odds, for the corners / cards / totals markets.
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS home_corners smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS away_corners smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS home_yellow  smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS away_yellow  smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS home_red     smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS away_red     smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS home_fouls   smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS away_fouls   smallint;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS referee      text;      -- Premier League only
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS odds_over25  real;      -- total goals > 2.5
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS odds_under25 real;
ALTER TABLE fd_matches ADD COLUMN IF NOT EXISTS ou_odds_source text;

-- Walk-forward predictions for the count markets (goals, corners, cards, shots).
CREATE TABLE IF NOT EXISTS market_predictions (
    model        text NOT NULL,                 -- poisson | league_average | market
    market       text NOT NULL,                 -- goals | corners | yellow | shots
    league_code  text NOT NULL,
    season       text NOT NULL,
    match_date   date NOT NULL,
    home_team    text NOT NULL,
    away_team    text NOT NULL,
    exp_home     real,
    exp_away     real,
    line         real NOT NULL,                 -- e.g. 2.5 goals, 9.5 corners
    p_over       real NOT NULL,
    PRIMARY KEY (model, market, league_code, season, home_team, away_team)
);

-- Full season schedule (fixturedownload.com), team names mapped to football-data's.
CREATE TABLE IF NOT EXISTS schedule (
    league_code text        NOT NULL,
    season      text        NOT NULL,
    round       smallint    NOT NULL,
    kickoff_utc timestamptz NOT NULL,
    home_team   text        NOT NULL,              -- football-data name
    away_team   text        NOT NULL,
    venue       text,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (league_code, season, home_team, away_team)
);

-- Alternative team names seen in other sources (schedule feed, StatsBomb), for OCR matching.
CREATE TABLE IF NOT EXISTS team_alias (
    league_code text NOT NULL,
    alias       text NOT NULL,
    team        text NOT NULL,                  -- football-data name
    source      text NOT NULL,
    PRIMARY KEY (league_code, alias)
);

-- API-Football player search responses, cached to stay far below the free daily quota.
CREATE TABLE IF NOT EXISTS api_player_cache (
    league_code text        NOT NULL,
    season      integer     NOT NULL,              -- season start year, API convention
    query       text        NOT NULL,
    payload     jsonb       NOT NULL,
    fetched_at  timestamptz NOT NULL,
    PRIMARY KEY (league_code, season, query)
);

-- A match is identified by its date too: Liga MX and Argentina repeat the same home/away
-- pairing within one season label (Apertura/Clausura, play-offs). Idempotent migration.
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['fd_matches', 'predictions', 'market_predictions'] LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.key_column_usage
            WHERE table_name = t AND constraint_name = t || '_pkey' AND column_name = 'match_date'
        ) THEN
            EXECUTE format('ALTER TABLE %I DROP CONSTRAINT %I', t, t || '_pkey');
            IF t = 'fd_matches' THEN
                ALTER TABLE fd_matches ADD PRIMARY KEY (league_code, season, home_team, away_team, match_date);
            ELSIF t = 'predictions' THEN
                ALTER TABLE predictions
                    ADD PRIMARY KEY (experiment, model, league_code, season, home_team, away_team, match_date);
            ELSE
                ALTER TABLE market_predictions
                    ADD PRIMARY KEY (model, market, league_code, season, home_team, away_team, match_date);
            END IF;
        END IF;
    END LOOP;
END $$;

-- ------------------------------------------------------------ API-Football (api-sports.io)
-- Parsed from /fixtures?ids responses; the raw JSON is kept gzipped on disk
-- (data/raw/api_football/) so new fields can be extracted without re-downloading.
CREATE TABLE IF NOT EXISTS api_fixture (
    fixture_id   integer PRIMARY KEY,
    league_code  text        NOT NULL,
    season       integer     NOT NULL,              -- API convention: start year
    kickoff_utc  timestamptz NOT NULL,
    round        text,
    status       text        NOT NULL,
    home_id      integer     NOT NULL,
    home_name    text        NOT NULL,
    away_id      integer     NOT NULL,
    away_name    text        NOT NULL,
    home_goals   smallint,
    away_goals   smallint,
    referee      text,
    fetched_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_api_fixture_league ON api_fixture (league_code, season, kickoff_utc);

CREATE TABLE IF NOT EXISTS api_team_match (
    fixture_id   integer  NOT NULL REFERENCES api_fixture ON DELETE CASCADE,
    team_id      integer  NOT NULL,
    is_home      boolean  NOT NULL,
    shots_on     smallint,
    shots_off    smallint,
    shots_total  smallint,
    shots_blocked smallint,
    shots_inside smallint,
    corners      smallint,
    fouls        smallint,
    offsides     smallint,
    yellow       smallint,
    red          smallint,
    saves        smallint,
    passes       smallint,
    possession   real,                              -- percent
    xg           real,                              -- expected_goals, when the API has it
    PRIMARY KEY (fixture_id, team_id)
);

CREATE TABLE IF NOT EXISTS api_player_match (
    fixture_id   integer  NOT NULL REFERENCES api_fixture ON DELETE CASCADE,
    player_id    integer  NOT NULL,
    team_id      integer  NOT NULL,
    player_name  text     NOT NULL,
    position     text,                              -- G / D / M / F
    minutes      smallint,
    substitute   boolean,
    started      boolean,                           -- in the starting XI (lineups)
    rating       real,
    shots_total  smallint,
    shots_on     smallint,
    goals        smallint,
    assists      smallint,
    key_passes   smallint,
    yellow       smallint,
    red          smallint,
    fouls_committed smallint,
    fouls_drawn  smallint,
    PRIMARY KEY (fixture_id, player_id)
);
CREATE INDEX IF NOT EXISTS ix_api_player_match_player ON api_player_match (player_id);

-- ---------------------------------------------------------------- lineups, injuries, upcoming

-- One row per team per fixture: formation and coach (from the stored /fixtures payload).
CREATE TABLE IF NOT EXISTS api_lineup (
    fixture_id   integer  NOT NULL REFERENCES api_fixture ON DELETE CASCADE,
    team_id      integer  NOT NULL,
    formation    text,
    coach        text,
    PRIMARY KEY (fixture_id, team_id)
);

-- Every player in the match sheet. grid = 'row:col' for starters: row 1 = goalkeeper,
-- col 1 = the team's left flank. Substitutes have no grid.
CREATE TABLE IF NOT EXISTS api_lineup_player (
    fixture_id   integer  NOT NULL REFERENCES api_fixture ON DELETE CASCADE,
    team_id      integer  NOT NULL,
    player_id    integer  NOT NULL,
    player_name  text     NOT NULL,
    number       smallint,
    pos          text,                              -- G / D / M / F
    grid         text,
    started      boolean  NOT NULL,
    PRIMARY KEY (fixture_id, player_id)
);
CREATE INDEX IF NOT EXISTS ix_api_lineup_player_team ON api_lineup_player (team_id);

-- Fixtures not played yet (kept apart from api_fixture, whose rows mean "downloaded").
CREATE TABLE IF NOT EXISTS api_upcoming (
    fixture_id   integer PRIMARY KEY,
    league_code  text        NOT NULL,
    season       integer     NOT NULL,
    kickoff_utc  timestamptz NOT NULL,
    round        text,
    home_id      integer     NOT NULL,
    home_name    text        NOT NULL,
    away_id      integer     NOT NULL,
    away_name    text        NOT NULL,
    fetched_at   timestamptz NOT NULL DEFAULT now()
);

-- Players reported out (or doubtful) for a fixture: /injuries?league&season.
CREATE TABLE IF NOT EXISTS api_injury (
    fixture_id   integer  NOT NULL,
    league_code  text     NOT NULL,
    team_id      integer  NOT NULL,
    player_id    integer  NOT NULL,
    player_name  text     NOT NULL,
    kind         text     NOT NULL,                 -- 'Missing Fixture' / 'Questionable'
    reason       text,
    kickoff_utc  timestamptz,
    fetched_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (fixture_id, player_id)
);
CREATE INDEX IF NOT EXISTS ix_api_injury_team ON api_injury (team_id, kickoff_utc);
