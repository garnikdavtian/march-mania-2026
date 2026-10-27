"""Sanity checks for the rating math. Run: python test_mania.py (or pytest)."""
import numpy as np
import pandas as pd

import mania

# A beats B by 10, B beats C by 10, A beats C by 20: a perfectly consistent season.
GAMES = pd.DataFrame({
    "Season": 2000, "DayNum": [1, 2, 3],
    "WTeamID": [1, 2, 1], "WScore": [80, 80, 90],
    "LTeamID": [2, 3, 3], "LScore": [70, 70, 70], "WLoc": "N",
})


def test_srs_recovers_consistent_ratings():
    r = mania.srs(GAMES).set_index("TeamID")["SRS"]
    assert np.allclose(r.loc[[1, 2, 3]], [10, 0, -10], atol=1e-6)


def test_elo_is_zero_sum_and_orders_teams():
    r = mania.elo(GAMES).set_index("TeamID")["Elo"]
    assert np.isclose(r.sum(), 3 * mania.ELO_BASE)
    assert r[1] > r[2] > r[3]


def test_matchups_are_antisymmetric():
    feats = pd.DataFrame({"Season": 2000, "TeamID": [1, 2], "Elo": [1600.0, 1400.0], "SRS": [5.0, -5.0], "ConfAbbrev": "x"})
    ab = mania.matchups(pd.DataFrame({"Season": [2000], "TeamA": [1], "TeamB": [2]}), feats)
    ba = mania.matchups(pd.DataFrame({"Season": [2000], "TeamA": [2], "TeamB": [1]}), feats)
    assert np.allclose(ab.filter(like="D_").values, -ba.filter(like="D_").values)
    assert np.isclose(mania.elo_prob(ab)[0] + mania.elo_prob(ba)[0], 1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all checks pass")
