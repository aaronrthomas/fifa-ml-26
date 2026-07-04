"""
generate_submission_direct.py
==============================
Standalone script that generates submission.csv WITHOUT requiring all
ML packages (no XGBoost, CatBoost etc needed).

Run with: python generate_submission_direct.py

This uses only numpy + scipy + pandas (already standard).
The full main.py requires all packages in requirements.txt.
"""

import os, sys, json, math
import numpy as np
from scipy.stats import poisson

# ── paths ──────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
LEAGUE_AVG    = 1.15   # WC knockout avg goals/team/90min
RHO           = -0.12  # Dixon-Coles correlation
N_SIMS        = 250_000

# ── Team data (pre-computed TSI inputs) ───────────────────────────────────
# Format: team -> {atk, def_, elo, form, sq_val, xg_diff, host, avail, pk}
TEAMS = {
    # elo normalised [0-1], form score, squad_val [0-1], etc.
    # attack/defense raw strengths for Poisson model
    "France":      {"atk":1.52,"def_":0.61,"elo_raw":2063,"form":0.88,"sq_val":1520,"pk":0.60,"host":False},
    "Argentina":   {"atk":1.48,"def_":0.68,"elo_raw":2044,"form":0.85,"sq_val": 880,"pk":0.80,"host":False},
    "Spain":       {"atk":1.61,"def_":0.53,"elo_raw":2038,"form":0.88,"sq_val":1220,"pk":0.57,"host":False},
    "Brazil":      {"atk":1.44,"def_":0.72,"elo_raw":2001,"form":0.80,"sq_val": 912,"pk":0.60,"host":False},
    "England":     {"atk":1.38,"def_":0.64,"elo_raw":1985,"form":0.83,"sq_val":1360,"pk":0.40,"host":False},
    "Portugal":    {"atk":1.41,"def_":0.71,"elo_raw":1978,"form":0.85,"sq_val":1010,"pk":0.67,"host":False},
    "Belgium":     {"atk":1.29,"def_":0.78,"elo_raw":1945,"form":0.73,"sq_val": 598,"pk":0.50,"host":False},
    "Netherlands": {"atk":1.33,"def_":0.75,"elo_raw":1942,"form":0.78,"sq_val": 841,"pk":0.38,"host":False},
    "Colombia":    {"atk":1.27,"def_":0.82,"elo_raw":1921,"form":0.84,"sq_val": 544,"pk":0.50,"host":False},
    "Morocco":     {"atk":1.08,"def_":0.69,"elo_raw":1918,"form":0.78,"sq_val": 341,"pk":0.50,"host":False},
    "Switzerland": {"atk":1.14,"def_":0.84,"elo_raw":1908,"form":0.79,"sq_val": 398,"pk":0.50,"host":False},
    "USA":         {"atk":1.21,"def_":0.88,"elo_raw":1878,"form":0.73,"sq_val": 389,"pk":0.50,"host":True},
    "Mexico":      {"atk":1.12,"def_":0.91,"elo_raw":1871,"form":0.72,"sq_val": 271,"pk":0.33,"host":True},
    "Canada":      {"atk":1.09,"def_":0.94,"elo_raw":1852,"form":0.76,"sq_val": 298,"pk":0.50,"host":True},
    "Norway":      {"atk":1.35,"def_":0.86,"elo_raw":1848,"form":0.78,"sq_val": 521,"pk":0.50,"host":False},
    "Egypt":       {"atk":1.06,"def_":0.87,"elo_raw":1789,"form":0.67,"sq_val": 182,"pk":0.50,"host":False},
    "Paraguay":    {"atk":1.04,"def_":0.93,"elo_raw":1761,"form":0.64,"sq_val": 134,"pk":0.50,"host":False},
}

# ── Goalscorer data (jersey number -> scoring weight) ─────────────────────
# top scorers per team: [(jersey, weight)]
SCORERS = {
    "France":      [(10,6.8),(9,3.4),(11,3.1),(7,2.8),(14,1.0),(22,0.5)],
    "Argentina":   [(10,5.5),(22,3.8),(9,4.2),(11,2.4),(7,1.4),(24,1.2)],
    "Spain":       [(19,3.4),(17,3.2),(7,2.8),(10,1.9),(16,1.1),(8,1.3)],
    "Brazil":      [(7,5.2),(9,3.5),(10,3.1),(11,3.8),(8,1.2)],
    "England":     [(9,5.2),(22,3.1),(7,3.4),(47,2.8),(10,2.9),(20,2.7)],
    "Portugal":    [(7,4.4),(9,3.8),(11,3.4),(8,2.2),(10,3.1)],
    "Belgium":     [(9,3.8),(11,3.3),(14,2.6),(7,1.7),(19,2.7)],
    "Netherlands": [(9,4.0),(11,3.2),(10,2.8),(8,1.5)],
    "Colombia":    [(7,3.2),(14,3.5),(10,1.8),(19,2.8)],
    "Morocco":     [(9,3.6),(7,2.4),(14,2.8),(2,0.8),(8,0.9)],
    "Switzerland": [(7,3.0),(25,2.6),(10,1.0),(11,2.2),(20,0.7)],
    "USA":         [(10,3.1),(12,3.3),(9,3.0),(21,2.2)],
    "Mexico":      [(9,2.8),(10,2.2),(7,1.9)],
    "Canada":      [(7,4.5),(9,3.1),(10,1.8),(11,2.6)],
    "Norway":      [(9,8.2),(12,3.8),(10,2.1),(23,1.0)],
    "Egypt":       [(11,5.6),(14,3.8),(9,3.1),(10,1.1)],
    "Paraguay":    [(9,3.4),(11,2.8),(7,1.9)],
}

# ── Core Poisson model ─────────────────────────────────────────────────────

def xg_for_fixture(ta, tb, host_a=False):
    a = TEAMS[ta]; b = TEAMS[tb]
    host_factor = 1.08 if host_a else 1.0
    xg_a = a["atk"] * b["def_"] * LEAGUE_AVG * host_factor
    xg_b = b["atk"] * a["def_"] * LEAGUE_AVG
    return round(max(xg_a, 0.20), 3), round(max(xg_b, 0.20), 3)

def score_matrix(xg_a, xg_b, maxg=7):
    g = np.arange(maxg+1)
    pa = poisson.pmf(g, xg_a); pb = poisson.pmf(g, xg_b)
    M = np.outer(pa, pb)
    # Dixon-Coles correction
    for i in range(min(2,maxg+1)):
        for j in range(min(2,maxg+1)):
            if i==0 and j==0: tau = 1 - xg_a*xg_b*RHO
            elif i==0 and j==1: tau = 1 + xg_a*RHO
            elif i==1 and j==0: tau = 1 + xg_b*RHO
            elif i==1 and j==1: tau = 1 - RHO
            else: tau = 1.0
            M[i,j] *= tau
    return M / M.sum()

def outcome_probs(M):
    p_a = float(np.sum(np.tril(M,-1)))
    p_d = float(np.sum(np.diag(M)))
    p_b = float(np.sum(np.triu(M,1)))
    t = p_a+p_d+p_b
    return p_a/t, p_d/t, p_b/t

def elo_prob(ea, eb):
    return 1/(1+10**((eb-ea)/400))

def predict_fixture(ta, tb):
    host_a = TEAMS.get(ta,{}).get("host",False)
    xg_a, xg_b = xg_for_fixture(ta, tb, host_a)
    M = score_matrix(xg_a, xg_b)
    p_a, p_d, p_b = outcome_probs(M)
    # KO probabilities — corrected: 100% of draws go to ET (KO tournament rules)
    et_prob = p_d * 1.0                 # all 90-min draws go to ET
    pk_prob = et_prob * 0.42            # ~42% of ET games end in penalties (historical WC rate)
    # Most likely score
    idx = np.unravel_index(np.argmax(M), M.shape)
    best_h, best_a = int(idx[0]), int(idx[1])
    # Top 5
    flat = np.argsort(M, axis=None)[::-1][:5]
    top5 = [(int(np.unravel_index(f,M.shape)[0]),
             int(np.unravel_index(f,M.shape)[1]),
             round(float(M[np.unravel_index(f,M.shape)]),4)) for f in flat]
    # Winner probability (total including ET/PK)
    et_h = et_prob*(1-0.5)*p_a/(p_a+p_b+1e-9)
    et_a = et_prob*(1-0.5)*p_b/(p_a+p_b+1e-9)
    pka = TEAMS.get(ta,{}).get("pk",0.5); pkb = TEAMS.get(tb,{}).get("pk",0.5)
    pk_a = pk_prob * pka/(pka+pkb+1e-9)
    pk_b = pk_prob * pkb/(pka+pkb+1e-9)
    total_a = p_a + et_h + pk_a
    total_b = p_b + et_a + pk_b
    tot = total_a + total_b
    total_a /= tot; total_b /= tot
    return {
        "xg_a": xg_a, "xg_b": xg_b,
        "p_a": round(total_a,4), "p_b": round(total_b,4),
        "p_d": round(p_d,4), "p_et": round(et_prob,4), "p_pk": round(pk_prob,4),
        "best_h": best_h, "best_a": best_a,
        "top5": top5,
        "winner": ta if total_a >= total_b else tb,
    }

def pick_jerseys(team, n_goals, seed_offset=0):
    if n_goals == 0: return ""
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    sc = SCORERS.get(team, [(9,1.0)])
    jerseys_list = [j for j,_ in sc]
    weights = np.array([w for _,w in sc], dtype=float)
    weights /= weights.sum()
    picks = rng.choice(jerseys_list, size=n_goals, p=weights, replace=True)
    return ";".join(str(p) for p in sorted(picks))

# ── Monte Carlo ────────────────────────────────────────────────────────────

def simulate_fixture_once(ta, tb, rng):
    xg_a, xg_b = xg_for_fixture(ta, tb, TEAMS.get(ta,{}).get("host",False))
    h = int(rng.poisson(xg_a)); a = int(rng.poisson(xg_b))
    if h > a: return ta
    elif a > h: return tb
    else:
        pka = TEAMS.get(ta,{}).get("pk",0.5); pkb = TEAMS.get(tb,{}).get("pk",0.5)
        return ta if rng.uniform() < pka/(pka+pkb+1e-9) else tb

def monte_carlo(qf_list, n=N_SIMS):
    rng = np.random.default_rng(RANDOM_SEED)
    counts = {t:{"qf":0,"sf":0,"fin":0,"champ":0} for t in TEAMS}
    for _ in range(n):
        # QF
        qw = [simulate_fixture_once(ta,tb,rng) for _,ta,tb,*__ in qf_list]
        for w in qw:
            if w in counts: counts[w]["qf"] += 1
        # SF
        sf1_teams = (qw[0],qw[1]) if len(qw)>1 else ("TBD","TBD")
        sf2_teams = (qw[2],qw[3]) if len(qw)>3 else ("TBD","TBD")
        sf_winners = []
        for ta,tb in [sf1_teams, sf2_teams]:
            if ta=="TBD" or tb=="TBD": continue
            w = simulate_fixture_once(ta,tb,rng)
            sf_winners.append(w)
            if w in counts: counts[w]["sf"] += 1
        # Final
        if len(sf_winners)==2:
            champion = simulate_fixture_once(sf_winners[0],sf_winners[1],rng)
            runner_up = sf_winners[1] if champion==sf_winners[0] else sf_winners[0]
            if champion  in counts: counts[champion]["fin"] += 1; counts[champion]["champ"] += 1
            if runner_up in counts: counts[runner_up]["fin"] += 1
    probs = {t:{k:v/n for k,v in d.items()} for t,d in counts.items()}
    return probs

# ── R16 → QF resolution ───────────────────────────────────────────────────

def resolve_r16():
    """Use Poisson model to predict each R16 winner."""
    r16 = [
        ("R16_1","Canada","Morocco"),
        ("R16_2","Paraguay","France"),
        ("R16_3","Brazil","Norway"),
        ("R16_4","Mexico","England"),
        ("R16_5","Portugal","Spain"),
        ("R16_6","USA","Belgium"),
        ("R16_7","Argentina","Egypt"),
        ("R16_8","Switzerland","Colombia"),
    ]
    winners = {}
    print("\nR16 Predictions:")
    for mid, ta, tb in r16:
        r = predict_fixture(ta, tb)
        winners[mid] = r["winner"]
        p = r["p_a"] if r["winner"]==ta else r["p_b"]
        print(f"  {mid}: {ta} vs {tb} → {r['winner']} ({p:.0%}) xG:{r['xg_a']:.2f}-{r['xg_b']:.2f}")
    return winners

def build_qf(r16_winners):
    bracket = [
        ("QF_001", r16_winners["R16_2"], r16_winners["R16_1"], "2026-07-09"),
        ("QF_002", r16_winners["R16_5"], r16_winners["R16_6"], "2026-07-10"),
        ("QF_003", r16_winners["R16_3"], r16_winners["R16_4"], "2026-07-11"),
        ("QF_004", r16_winners["R16_7"], r16_winners["R16_8"], "2026-07-12"),
    ]
    return bracket

# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("="*62)
    print(" FIFA WORLD CUP 2026 — KNOCKOUT PREDICTOR (Standalone)")
    print("="*62)

    r16w = resolve_r16()
    qf_list = build_qf(r16w)

    print("\nQF Fixtures:")
    for fid,ta,tb,dt in qf_list:
        print(f"  {fid}: {ta} vs {tb}")

    # Run Monte Carlo
    print(f"\nRunning {N_SIMS:,} Monte Carlo simulations...")
    mc_probs = monte_carlo(qf_list, n=N_SIMS)

    # Build all match predictions
    rows = []
    qf_results = {}
    qf_winners = []

    print("\n── Quarter Finals ──")
    for fid, ta, tb, dt in qf_list:
        r = predict_fixture(ta, tb)
        h_s, a_s = r["best_h"], r["best_a"]
        winner = r["winner"]
        is_pk = r["p_pk"] > 0.15
        is_et = r["p_et"] > 0.25

        # If draw score → ET resolution
        if h_s == a_s:
            if winner == ta: h_s += 1
            elif not is_pk: a_s += 1
            is_et = True

        jerseys_h = pick_jerseys(ta, h_s, seed_offset=hash(fid+"h")%1000)
        jerseys_a = pick_jerseys(tb, a_s, seed_offset=hash(fid+"a")%1000)

        row = {
            "match_id": fid, "stage": "Quarter Final",
            "home_team": ta, "away_team": tb,
            "predicted_home_score": h_s, "predicted_away_score": a_s,
            "predicted_scorers_home": jerseys_h,
            "predicted_scorers_away": jerseys_a,
            "predicted_winner": winner,
        }
        rows.append(row)
        qf_winners.append(winner)
        print(f"  {fid}: {ta} {h_s}-{a_s} {tb}  → {winner}  "
              f"(P={r['p_a']:.0%}/{r['p_b']:.0%}, ET:{r['p_et']:.0%}, PK:{r['p_pk']:.0%})")
        print(f"         Scorers: {ta}:{jerseys_h or '—'}  {tb}:{jerseys_a or '—'}")

    # SF
    sf_winners = []; sf_losers = []
    sf_fixtures = [
        ("SF_001", qf_winners[0], qf_winners[1], "2026-07-14"),
        ("SF_002", qf_winners[2], qf_winners[3], "2026-07-15"),
    ]
    print("\n── Semi Finals ──")
    for fid, ta, tb, dt in sf_fixtures:
        r = predict_fixture(ta, tb)
        h_s, a_s = r["best_h"], r["best_a"]
        winner = r["winner"]
        loser  = tb if winner==ta else ta
        is_pk  = r["p_pk"] > 0.15
        if h_s == a_s:
            if winner==ta: h_s += 1
            elif not is_pk: a_s += 1
        sf_winners.append(winner); sf_losers.append(loser)
        jerseys_h = pick_jerseys(ta, h_s, seed_offset=hash(fid+"h")%1000)
        jerseys_a = pick_jerseys(tb, a_s, seed_offset=hash(fid+"a")%1000)
        row = {
            "match_id": fid, "stage": "Semi Final",
            "home_team": ta, "away_team": tb,
            "predicted_home_score": h_s, "predicted_away_score": a_s,
            "predicted_scorers_home": jerseys_h,
            "predicted_scorers_away": jerseys_a,
            "predicted_winner": winner,
        }
        rows.append(row)
        print(f"  {fid}: {ta} {h_s}-{a_s} {tb}  → {winner}")
        print(f"         Scorers: {ta}:{jerseys_h or '—'}  {tb}:{jerseys_a or '—'}")

    # Third Place
    if len(sf_losers)==2:
        ta3,tb3 = sf_losers[0], sf_losers[1]
        r = predict_fixture(ta3, tb3)
        h_s, a_s = r["best_h"], r["best_a"]
        winner = r["winner"]
        if h_s==a_s:
            if winner==ta3: h_s+=1
            else: a_s+=1
        jerseys_h = pick_jerseys(ta3, h_s, seed_offset=333)
        jerseys_a = pick_jerseys(tb3, a_s, seed_offset=444)
        row = {
            "match_id": "TP_001", "stage": "Third Place",
            "home_team": ta3, "away_team": tb3,
            "predicted_home_score": h_s, "predicted_away_score": a_s,
            "predicted_scorers_home": jerseys_h,
            "predicted_scorers_away": jerseys_a,
            "predicted_winner": winner,
        }
        rows.append(row)
        print(f"\n── Third Place ──")
        print(f"  TP_001: {ta3} {h_s}-{a_s} {tb3}  → {winner}")

    # Final
    if len(sf_winners)==2:
        fa,fb = sf_winners[0], sf_winners[1]
        r = predict_fixture(fa, fb)
        h_s, a_s = r["best_h"], r["best_a"]
        winner = r["winner"]
        is_pk  = r["p_pk"] > 0.15
        if h_s==a_s:
            if winner==fa: h_s+=1
            elif not is_pk: a_s+=1
        jerseys_h = pick_jerseys(fa, h_s, seed_offset=999)
        jerseys_a = pick_jerseys(fb, a_s, seed_offset=888)
        row = {
            "match_id": "F_001", "stage": "Final",
            "home_team": fa, "away_team": fb,
            "predicted_home_score": h_s, "predicted_away_score": a_s,
            "predicted_scorers_home": jerseys_h,
            "predicted_scorers_away": jerseys_a,
            "predicted_winner": winner,
        }
        rows.append(row)
        print(f"\n── Final ──")
        print(f"  F_001: {fa} {h_s}-{a_s} {fb}  → 🏆 {winner}")
        print(f"         Scorers: {fa}:{jerseys_h or '—'}  {fb}:{jerseys_a or '—'}")

    # Write submission.csv
    import csv
    cols = ["match_id","stage","home_team","away_team",
            "predicted_home_score","predicted_away_score",
            "predicted_scorers_home","predicted_scorers_away",
            "predicted_winner"]
    out_path = os.path.join(PRED_DIR, "submission.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n✅  submission.csv saved → {out_path}")

    # Champion probability summary
    champ_sorted = sorted(mc_probs.items(), key=lambda x: -x[1]["champ"])
    print("\n── Champion Probabilities (250k simulations) ──")
    for t, p in champ_sorted:
        if p["champ"] > 0.005:
            print(f"  {t:<15} {p['champ']:.1%}  (SF:{p['sf']:.0%}, Final:{p['fin']:.0%})")

    print(f"\n🏆 PREDICTED CHAMPION: {rows[-1]['predicted_winner']}")
    print("="*62)

if __name__ == "__main__":
    main()
