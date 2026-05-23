# StrokeLab

OTI tennis stroke analysis — powered by MediaPipe pose estimation and Claude AI coaching.

## What it does

Record or upload a tennis video and get instant feedback on all three OTI power sources:

| Power Source | What's measured |
|---|---|
| **Legs** | Knee bend depth during loading phase |
| **Shoulders** | X-factor (shoulder-hip separation) and rotation speed |
| **Late hit** | Contact point position relative to front hip |

Also reports: kinetic chain timing, swing direction (inside-out vs neutral vs outside-in).

## Structure

```
stroke-lab/
├── backend/    Python 3.11 · FastAPI · MediaPipe · Anthropic
└── mobile/     React Native · Expo
```

## Running locally

### Backend

```bash
cd backend
source venv/bin/activate
python main.py          # starts on http://localhost:8000
```

### Mobile

```bash
cd mobile
npx expo start
# then open Expo Go on your phone and scan the QR code
```

Set the `BASE_URL` in `mobile/src/api/client.ts` to your machine's local IP (e.g. `http://192.168.1.x:8000/api`) when testing on a physical device.

## Environment

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
