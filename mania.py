"""March Machine Learning Mania 2026 - 29th / 3,462 (Brier 0.1209).

Team-season ratings (Elo, SRS, Massey, box-score efficiency) -> A-minus-B
differentials -> logistic regression + gradient boosting + raw Elo, blended by
out-of-fold Brier. Run `python mania.py` to write submission.csv.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

DATA_DIR = Path(__file__).parent / "data"
ELO_BASE, ELO_K, ELO_HCA, ELO_CARRY = 1500.0, 20.0, 65.0, 0.75
CLIP = (0.01, 0.99)
RANDOM_STATE = 42

# lbfgs on unscaled features stops at max_iter. That early stop regularizes:
# standardizing the inputs measured worse in season-grouped CV (0.1829 vs 0.1822 men).
warnings.filterwarnings("ignore", category=ConvergenceWarning)

BOX = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk"]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def to_long(games: pd.DataFrame, box: bool = False) -> pd.DataFrame:
    """One row per team per game (winner/loser format -> team/opponent format)."""
    sides = []
    for me, opp, win in (("W", "L", 1), ("L", "W", 0)):
        d = pd.DataFrame({
            "Season": games["Season"], "DayNum": games["DayNum"],
            "TeamID": games[f"{me}TeamID"], "OppID": games[f"{opp}TeamID"],
            "ScoreFor": games[f"{me}Score"], "ScoreAgainst": games[f"{opp}Score"], "Win": win,
        })
        if box:
            for c in BOX:
                d[c] = games[me + c]
            for c in ("OR", "DR", "FGA"):
                d[f"Opp_{c}"] = games[opp + c]
        sides.append(d)
    df = pd.concat(sides, ignore_index=True)
    df["Margin"] = df["ScoreFor"] - df["ScoreAgainst"]
    if box:
        df["Poss"] = df["FGA"] - df["OR"] + df["TO"] + 0.475 * df["FTA"]
    return df


def season_basics(long: pd.DataFrame) -> pd.DataFrame:
    return long.groupby(["Season", "TeamID"], as_index=False).agg(
        Games=("Win", "size"), Wins=("Win", "sum"), WinPct=("Win", "mean"),
        PF=("ScoreFor", "sum"), PA=("ScoreAgainst", "sum"), Margin=("Margin", "mean"),
        PFpg=("ScoreFor", "mean"), PApg=("ScoreAgainst", "mean"),
    )


def season_efficiency(long: pd.DataFrame) -> pd.DataFrame:
    """Tempo-free box-score rates (Dean Oliver's four factors and friends)."""
    cols = BOX + ["Opp_OR", "Opp_DR", "Opp_FGA", "Poss", "ScoreFor", "ScoreAgainst"]
    a = long.groupby(["Season", "TeamID"], as_index=False)[cols].sum()
    a["Games"] = long.groupby(["Season", "TeamID"]).size().values
    rate = lambda num, den: num / den.clip(lower=1)
    return a[["Season", "TeamID"]].assign(
        OffEff=100 * rate(a["ScoreFor"], a["Poss"]),
        DefEff=100 * rate(a["ScoreAgainst"], a["Poss"]),
        Tempo=rate(a["Poss"], a["Games"]),
        eFG=rate(a["FGM"] + 0.5 * a["FGM3"], a["FGA"]),
        TOV=rate(a["TO"], a["Poss"]),
        ORB=rate(a["OR"], a["OR"] + a["Opp_DR"]),
        FTR=rate(a["FTA"], a["FGA"]),
        **{"3PAr": rate(a["FGA3"], a["FGA"])},
        AST=rate(a["Ast"], a["FGM"]),
        StlRate=rate(a["Stl"], a["Poss"]),
        BlkRate=rate(a["Blk"], a["Opp_FGA"]),
    )


def srs(compact: pd.DataFrame, iters: int = 100) -> pd.DataFrame:
    """Simple Rating System: rating = mean over games of (margin + opponent rating)."""
    rows = []
    for season, g in to_long(compact).groupby("Season"):
        teams, team_idx = np.unique(g["TeamID"].values, return_inverse=True)
        opp_idx = np.searchsorted(teams, g["OppID"].values)
        n = np.bincount(team_idx)
        margin = g["Margin"].values.astype(float)
        r = np.bincount(team_idx, margin) / n
        for _ in range(iters):
            r = np.bincount(team_idx, margin + r[opp_idx]) / n
        rows.append(pd.DataFrame({"Season": season, "TeamID": teams, "SRS": r}))
    return pd.concat(rows, ignore_index=True)


def elo(compact: pd.DataFrame) -> pd.DataFrame:
    """Margin-aware Elo with home court and 75% carry-over between seasons."""
    compact = compact.sort_values(["Season", "DayNum"])
    last: dict[int, float] = {}
    rows = []
    for season, g in compact.groupby("Season", sort=True):
        teams = pd.unique(g[["WTeamID", "LTeamID"]].values.ravel())
        r = {t: ELO_CARRY * last.get(t, ELO_BASE) + (1 - ELO_CARRY) * ELO_BASE for t in teams}
        for w, l, ws, ls, loc in zip(g["WTeamID"], g["LTeamID"], g["WScore"], g["LScore"], g["WLoc"]):
            hca = {"H": ELO_HCA, "A": -ELO_HCA}.get(loc, 0.0)
            exp_w = 1 / (1 + 10 ** ((r[l] - r[w] - hca) / 400))
            gap = r[w] + hca - r[l]
            delta = ELO_K * math.log(max(ws - ls, 1) + 1) * 2.2 / (abs(gap) * 0.001 + 2.2) * (1 - exp_w)
            r[w] += delta
            r[l] -= delta
        rows += [{"Season": season, "TeamID": int(t), "Elo": r[t]} for t in teams]
        last.update(r)
    return pd.DataFrame(rows)


def recent_form(compact: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    long = to_long(compact).sort_values(["Season", "TeamID", "DayNum"])
    return long.groupby(["Season", "TeamID"]).tail(n).groupby(["Season", "TeamID"], as_index=False).agg(
        RecentWinPct=("Win", "mean"), RecentMargin=("Margin", "mean"))


def massey_composite() -> pd.DataFrame:
    """Mean percentile rank over every Massey system's last pre-tournament ranking (men only)."""
    df = pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv", dtype={"SystemName": "category"})
    df = df[df["RankingDayNum"] <= 133]
    last_day = df.groupby(["Season", "SystemName"], observed=True)["RankingDayNum"].transform("max")
    df = df[df["RankingDayNum"] == last_day]
    max_rank = df.groupby(["Season", "SystemName"], observed=True)["OrdinalRank"].transform("max")
    df = df.assign(RankPct=df["OrdinalRank"] / max_rank.clip(lower=1))
    comp = df.groupby(["Season", "TeamID"], as_index=False)["RankPct"].mean()
    return comp.assign(Massey=1 - comp["RankPct"])[["Season", "TeamID", "Massey"]]


def team_features(prefix: str) -> pd.DataFrame:
    """One row per (Season, TeamID) with every rating and season stat."""
    compact = read(f"{prefix}RegularSeasonCompactResults")
    detailed = read(f"{prefix}RegularSeasonDetailedResults")
    key = ["Season", "TeamID"]
    f = season_basics(to_long(compact))
    for part in (season_efficiency(to_long(detailed, box=True)), srs(compact), elo(compact), recent_form(compact)):
        f = f.merge(part, on=key, how="left")
    f = f.merge(read(f"{prefix}TeamConferences"), on=key, how="left")
    f["ConfSRS"] = f.groupby(["Season", "ConfAbbrev"])["SRS"].transform("mean")
    f["SRSRankPct"] = f.groupby("Season")["SRS"].rank(pct=True)
    f["EloRankPct"] = f.groupby("Season")["Elo"].rank(pct=True)
    if prefix == "M":
        f = f.merge(massey_composite(), on=key, how="left")
    # Box scores start in 2003 (M) / 2010 (W): fill gaps with the season mean, then the global mean.
    cols = f.columns.difference(key + ["ConfAbbrev"])
    f[cols] = f[cols].fillna(f.groupby("Season")[cols].transform("mean"))
    f[cols] = f[cols].fillna(f[cols].mean())
    return f


def matchups(pairs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Season/TeamA/TeamB rows -> D_<feature> = A minus B, plus A_Elo/B_Elo for the raw Elo signal."""
    feats = features.drop(columns="ConfAbbrev")
    cols = [c for c in feats.columns if c not in ("Season", "TeamID")]
    side = lambda t: pairs.merge(feats, left_on=["Season", t], right_on=["Season", "TeamID"], how="left", validate="m:1")
    a, b = side("TeamA"), side("TeamB")
    out = pd.DataFrame({f"D_{c}": a[c].values - b[c].values for c in cols}, index=pairs.index)
    out["A_Elo"], out["B_Elo"] = a["Elo"].values, b["Elo"].values
    return out


def tourney_games(prefix: str, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    games = read(f"{prefix}NCAATourneyCompactResults")
    pairs = pd.DataFrame({
        "Season": games["Season"],
        "TeamA": games[["WTeamID", "LTeamID"]].min(axis=1),
        "TeamB": games[["WTeamID", "LTeamID"]].max(axis=1),
    })
    y = (pairs["TeamA"] == games["WTeamID"]).astype(int)
    m = matchups(pairs, features)
    return m.filter(like="D_"), y, pairs["Season"], m


def elo_prob(m: pd.DataFrame) -> np.ndarray:
    return 1 / (1 + 10 ** ((m["B_Elo"].values - m["A_Elo"].values) / 400))


def new_models():
    return (LogisticRegression(max_iter=500, C=1.0),
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                           l2_regularization=0.1, random_state=RANDOM_STATE))


def fit(X: pd.DataFrame, y: pd.Series, groups: pd.Series, m: pd.DataFrame) -> dict:
    """Season-grouped 5-fold OOF -> grid-search blend weights -> refit both models on all games."""
    logit, hgb = new_models()
    oof = pd.DataFrame({"logit": 0.0, "hgb": 0.0, "elo": elo_prob(m)}, index=X.index)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        oof.iloc[va, 0] = logit.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
        oof.iloc[va, 1] = hgb.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]

    grid = [(w1, w2, 1 - w1 - w2) for w1 in np.arange(0, 1.01, 0.05) for w2 in np.arange(0, 1.01 - w1, 0.05)]
    weights = min(grid, key=lambda w: brier_score_loss(y, oof.values @ np.array(w)))
    oof["blend"] = oof[["logit", "hgb", "elo"]].values @ np.array(weights)
    return {"logit": logit.fit(X, y), "hgb": hgb.fit(X, y), "weights": weights, "oof": oof}


def predict(model: dict, X: pd.DataFrame, m: pd.DataFrame) -> np.ndarray:
    p = np.column_stack([model["logit"].predict_proba(X)[:, 1], model["hgb"].predict_proba(X)[:, 1], elo_prob(m)])
    return np.clip(p @ np.array(model["weights"]), *CLIP)


def main() -> None:
    sub = read("SampleSubmissionStage2")
    ids = sub["ID"].str.split("_", expand=True).astype(int)
    pairs = pd.DataFrame({"Season": ids[0], "TeamA": ids[1], "TeamB": ids[2]})
    for prefix, mask in (("M", pairs["TeamA"] < 3000), ("W", pairs["TeamA"] >= 3000)):
        features = team_features(prefix)
        X, y, groups, m = tourney_games(prefix, features)
        model = fit(X, y, groups, m)
        print(f"{prefix}: OOF Brier {brier_score_loss(y, model['oof']['blend']):.4f}, "
              f"weights logit/hgb/elo = {tuple(round(float(w), 2) for w in model['weights'])}")
        pm = matchups(pairs[mask], features)
        sub.loc[mask, "Pred"] = predict(model, pm.filter(like="D_"), pm)
    sub.to_csv("submission.csv", index=False)
    print(f"wrote submission.csv ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
