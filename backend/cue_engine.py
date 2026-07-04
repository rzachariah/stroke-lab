"""
PHASE 0 — turn per-stroke OTI metrics into spoken coaching cues, with a
repetition/novelty policy so it coaches instead of nagging.

Reads segment_run.py's per-stroke output, runs a stateful "coach" over the stroke
sequence, prints the transcript (SPEAK vs quiet + why), and speaks the cues with
macOS `say`. Tested offline on recorded clips before any live infra.

Coaching policy (the point of this iteration):
  - Coach ONE focus at a time = the weakest power source that's a fault.
  - Say it ONCE, then go quiet and let the player work on it (cooldown).
  - CONFIRM briefly when the focus improves past "good", then move on.
  - RE-CUE if the focus regresses, or REMIND gently if it persists a long time.
  - SWITCH focus only when a different fault clearly dominates.
  - Silence is the default — most reps say nothing.

Caveat: LATE-HIT assumes a struck ball; shadow/contact labels aren't wired in yet
(see ball_contact_probe.py), so late cues are marked [contact?].

Run:  source venv/bin/activate && python cue_engine.py [--limit N] [--no-say]
"""
import argparse, json, subprocess, time

STROKES = "inspect_out/My_recorded_video_382.MP4.strokes.json"

GOOD = 6.5        # score >= GOOD: that power source is not a fault
COOLDOWN = 4      # strokes to stay quiet on a dim after cueing it
REGRESS = 2.0     # score drop on the focus that triggers a re-cue
REMIND = 8        # strokes of persistent fault before a gentle reminder
SWITCH = 2.0      # how much worse a new dim must be to steal focus

CORRECT = {"legs": "Bend your knees more.",
           "shoulders": "Turn your shoulders more.",
           "late": "Catch it further out front."}
AGAIN = {"legs": "Still — sit into those legs.",
         "shoulders": "Still — more shoulder turn.",
         "late": "Still — get it out in front."}
CONFIRM = {"legs": "Better — good leg drive.",
           "shoulders": "Better — hold that turn.",
           "late": "Better — nice contact out front."}


def plausible(r):
    xf, kn, lt = r["peak_x_factor"], r["min_knee_bend"], r["contact_wrist_x_rel"]
    if xf is not None and (xf >= 89 or xf < 8):
        return False
    if kn is None or kn < 90:
        return False
    if lt is not None and lt < -0.03:
        return False
    return True


def dims_of(r):
    d = {}
    if r["leg_score"] is not None:
        d["legs"] = r["leg_score"]
    if r["shoulder_score"] is not None:
        d["shoulders"] = r["shoulder_score"]
    if r["late_hit_score"] is not None:
        d["late"] = r["late_hit_score"]
    return d


class Coach:
    def __init__(self):
        self.focus = None
        self.said_sc = {}      # dim -> score when we last cued it
        self.said_at = {}      # dim -> stroke index we last cued it

    def _cue(self, dim, sc, i, text):
        self.said_sc[dim] = sc
        self.said_at[dim] = i
        return text

    def step(self, i, dims):
        if not dims:
            return None, "no reliable metrics"
        weak = min(dims, key=dims.get)
        weak_sc = dims[weak]

        # --- we're currently coaching a focus that is measured this stroke ---
        if self.focus in dims:
            f_sc = dims[self.focus]
            f = self.focus
            if f_sc >= GOOD:                                   # improved -> confirm, drop it
                self.focus = None
                self.said_at.pop(f, None)
                return CONFIRM[f], f"{f} improved to {f_sc} -> confirm"
            if f_sc <= self.said_sc.get(f, 10) - REGRESS:      # regressed -> re-cue
                return self._cue(f, f_sc, i, AGAIN[f]), f"{f} regressed to {f_sc} -> re-cue"
            since = i - self.said_at.get(f, i)
            if (weak != f and weak_sc < GOOD and weak_sc <= f_sc - SWITCH
                    and since >= COOLDOWN):                    # new dominant fault -> switch
                self.focus = weak
                return self._cue(weak, weak_sc, i, CORRECT[weak]), \
                    f"{weak} ({weak_sc}) now dominates {f} ({f_sc}) -> switch"
            if since >= REMIND:                                # persistent -> gentle reminder
                return self._cue(f, f_sc, i, AGAIN[f]), f"{f} persists {since} strokes -> remind"
            return None, f"working on {f} (quiet {since}/{COOLDOWN})"

        # --- no active focus: adopt the weakest fault if it's off cooldown ---
        if weak_sc < GOOD:
            since = i - self.said_at.get(weak, -99)
            if weak in self.said_at and since < COOLDOWN:
                return None, f"{weak} weak but on cooldown ({since}/{COOLDOWN})"
            self.focus = weak
            return self._cue(weak, weak_sc, i, CORRECT[weak]), f"new focus {weak} ({weak_sc})"
        return None, "all power sources solid -> quiet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=99, help="max cues to SPEAK")
    ap.add_argument("--no-say", action="store_true", help="print only, no audio")
    args = ap.parse_args()

    data = json.load(open(STROKES))
    coach = Coach()
    spoken = 0
    n_strokes = n_spoke = 0
    print(f"{'#':>3} {'t':>6} {'act':>5}  cue / reason")
    for i, w in enumerate(data):
        r = w["report"]
        t = w["window"]["peak_time"]
        if not plausible(r):
            print(f"{i:>3} {t:>6.1f} {'skip':>5}  (non-stroke)")
            continue
        n_strokes += 1
        cue, why = coach.step(i, dims_of(r))
        if cue is None:
            print(f"{i:>3} {t:>6.1f} {'--':>5}  quiet · {why}")
            continue
        n_spoke += 1
        tag = " [contact?]" if "late" in why else ""
        print(f"{i:>3} {t:>6.1f} {'SAY':>5}  \"{cue}\"{tag}  · {why}")
        if not args.no_say and spoken < args.limit:
            subprocess.run(["say", cue]); time.sleep(0.35); spoken += 1

    print(f"\n{n_strokes} strokes -> {n_spoke} spoken ({100*n_spoke//max(1,n_strokes)}%), "
          f"{n_strokes - n_spoke} silent." + ("" if args.no_say else f"  Spoke {spoken}."))


if __name__ == "__main__":
    main()
