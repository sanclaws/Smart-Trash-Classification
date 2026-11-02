# Smart Trash AI — Vercel deployment files

This folder contains the runtime files needed to deploy the existing Flask + TensorFlow app on Vercel.

## Files included
- `app.py`: Flask entrypoint and image prediction route.
- `requirements.txt`: Python dependencies.
- `.python-version`: pins the Vercel runtime to Python 3.12.
- `vercel.json`: sets the Flask function maximum duration to 300 seconds.
- `trash_model.keras` and `classes.npy`: existing trained model and class labels.
- `templates/`: the two HTML pages.
- `public/style.css`: CSS served by Vercel from `/style.css`.

## Important behavior changes for Vercel
- Uploaded images are processed in memory and previewed as a data URI. They are not saved to `static/uploads`, because local function storage is not persistent.
- Static CSS is served from `public/style.css`, not Flask's `static` route.
- Maximum upload size is 8 MB.

## Deploy
Import the GitHub repository into Vercel and use the repository root as the Root Directory. Vercel should detect Flask from `requirements.txt` and the `app` object in `app.py`. Do not set a separate Build Command or Output Directory unless Vercel specifically asks for one.

The model was saved with Keras 3.14.1. Test a few known images after deployment and compare the predictions against the original local application.
