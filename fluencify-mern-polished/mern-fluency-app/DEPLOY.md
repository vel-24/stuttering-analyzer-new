# Deploying the Fluency App on free hosting

Four pieces, deployed in this order (each needs the address of the one before it):

1. MongoDB Atlas  ->  connection string
2. Hugging Face Space (Python inference)  ->  `https://<user>-fluency-inference.hf.space`
3. Render (Express API)  ->  `https://<name>.onrender.com`
4. Vercel (React client)  ->  your public link

Paths below are relative to `fluencify-mern-polished/mern-fluency-app/` in your repo.

## 0. Apply the kit to your repo

Copy these files from the kit into the same places in your repo, then commit and push:

| Kit file | Goes to |
|---|---|
| `inference-service/main.py` | replaces the existing file (adds an API-key check) |
| `inference-service/Dockerfile`, `.dockerignore`, `SPACE_README.md` | new files |
| `server/src/routes/analyze.js` | replaces the existing file (sends the key, longer timeout) |
| `server/src/index.js` | replaces the existing file (rate-limits /api/analyze) |
| `client/vercel.json` | new file (lets page refreshes work) |

**Python versions are pinned.** `inference-service/requirements.txt` pins the versions from the
environment where the app was tested (numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, xgboost 3.2.0,
joblib 1.6.0, librosa 0.11.0, torch 2.14.1, transformers 5.20.1), because the saved `.joblib` models
can fail to load or change results under different versions.

## 1. MongoDB Atlas (free M0)

1. Create a free cluster at mongodb.com/atlas.
2. Database Access: create a user with a **long random password**.
3. Network Access: add `0.0.0.0/0` (Render's free tier has no fixed IP). The password is then your protection, so make it strong.
4. Connect -> Drivers: copy the connection string and add a database name, e.g.
   `mongodb+srv://USER:PASSWORD@cluster0.xxxxx.mongodb.net/fluency_app?retryWrites=true&w=majority`

## 2. Hugging Face Space: the inference service

1. huggingface.co -> New Space. SDK: **Docker**, hardware: **CPU basic (free)**. Name it `fluency-inference`.
2. In your terminal (install Git LFS first: https://git-lfs.com):

```
git clone https://huggingface.co/spaces/<your-hf-username>/fluency-inference
cd fluency-inference
git lfs install
git lfs track "*.pt" "*.joblib"
# copy EVERYTHING from your repo's inference-service/ folder into this folder
# (including models/, data/whisper_ft_seed0.pt, Dockerfile, .dockerignore)
copy SPACE_README.md README.md      # Space needs this exact name; it holds the Space settings
git add .
git commit -m "Inference service"
git push
```

   When Git asks for a password, use a Hugging Face **write access token** (Settings -> Access Tokens).
3. Space -> Settings -> **Variables and secrets** -> new secret:
   `INFERENCE_API_KEY` = a long random string (save it; Render needs the same value).
4. Wait for the build (about 10-15 minutes the first time). Then open
   `https://<your-hf-username>-fluency-inference.hf.space/health`. You should see
   `"status":"ok"` and `modelsReady` showing `true` for each model.

Free Spaces go to sleep after inactivity and wake on the next request (about a minute).

## 3. Render: the Express API

1. render.com -> New -> Web Service -> connect your GitHub repo `vel-24/stuttering-analyzer-new`.
2. Settings:
   - Root Directory: `fluencify-mern-polished/mern-fluency-app/server`
   - Build Command: `npm install`
   - Start Command: `npm start`
   - Instance type: Free
3. Environment variables:

| Name | Value |
|---|---|
| `MONGO_URI` | your Atlas connection string |
| `JWT_SECRET` | a long random string (do not reuse the example value) |
| `JWT_EXPIRES_IN` | `7d` |
| `INFERENCE_SERVICE_URL` | `https://<your-hf-username>-fluency-inference.hf.space` (no trailing slash) |
| `INFERENCE_API_KEY` | the same value you set on the Space |
| `CLIENT_ORIGIN` | your Vercel address (fill in after step 4, then redeploy) |

4. Open `https://<name>.onrender.com/api/health`. It should return `{"status":"ok"}`.

The free tier sleeps after about 15 minutes idle, so the first request after a break is slow.

## 4. Vercel: the React client

1. vercel.com -> Add New -> Project -> import the same GitHub repo.
2. Settings:
   - Root Directory: `fluencify-mern-polished/mern-fluency-app/client`
   - Framework: Vite (auto-detected)
3. Environment variable: `VITE_API_BASE_URL` = `https://<name>.onrender.com/api`
4. Deploy. Your public link is the Vercel address (you can rename it under Settings -> Domains).
5. Go back to Render and set `CLIENT_ORIGIN` to exactly that address (e.g. `https://fluency-app.vercel.app`, no trailing slash), then redeploy.

## 5. Test

1. Open the Vercel link, sign up, record a few seconds with the microphone, and press Analyze.
2. The first analysis can take a minute or two if the Space or Render was asleep.
3. Check the results page does **not** show the "Demo result" warning. If it does, the inference service is not finding its model files; open its `/health` page.

## Common problems

| Symptom | Likely cause |
|---|---|
| Network or CORS error in the browser | `CLIENT_ORIGIN` on Render doesn't exactly match the Vercel address (https, no trailing slash). |
| "Analysis failed" / 502 | The Space is asleep or still building, or `INFERENCE_API_KEY` differs between Render and the Space. |
| 401 from the inference service | Same key mismatch. |
| Demo result warning | Model files missing in the Space, or `.joblib` version mismatch (see "Pin the Python versions"). |
| Microphone does not work | Page isn't on HTTPS or the browser blocked the permission. Vercel links are HTTPS. |
| Space build runs out of memory or time | Check the build log; pin smaller versions or remove unused packages from `requirements.txt`. |

## Before you share the link

- Anyone with the link can create an account and upload audio. Add a short note on the sign-up page that recordings are analyzed but the audio itself is not stored, and that results are saved to the account.
- This is a research prototype, not a medical device. Keep the "not a clinical diagnosis" notice visible.
- Never commit `.env` files or real secrets. Only the `.env.example` placeholders belong in the repo.
- Free tiers have limits and may change; for a long-lived deployment, check each provider's current free-plan terms.
