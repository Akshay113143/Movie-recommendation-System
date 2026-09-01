# Pushing this update to GitHub

Your repo currently has the 12 original files flat at the root. This update
restructures them into packages **and** adds the autoencoder + Streamlit app.
Run these from inside your local repo folder (`Downloads/files (8)`).

## Step 1 — replace the contents

Delete the old flat files and copy in everything from this download:

```bash
cd "/Users/akshaykumar/Downloads/files (8)"

# remove the old flat copies (git will record these as moves)
rm -f collaborative.py content.py data.py evaluate.py hybrid.py \
      analysis.py recommend.py run_pipeline.py README.md \
      demo_output.txt evaluation.png metrics.json

# copy everything from the new download into this folder, then:
ls
# you should see: recsys/  app/  results/  .streamlit/  streamlit_app.py
#                 run_pipeline.py  train_autoencoder.py  export_serving_bundle.py
#                 recommend.py  analysis.py  README.md  DEPLOYMENT.md
#                 requirements.txt  .gitignore
```

Hidden files (`.gitignore`, `.streamlit/`) may not show in Finder — press
`Cmd+Shift+.` to reveal them, and make sure both get copied.

## Step 2 — fix your git identity (one time)

Your first commit was attributed to `akshaykumar@akshaykumars-MacBook-Air.local`,
which GitHub cannot link to your profile, so it will not appear on your
contribution graph.

```bash
git config --global user.name "Akshay Kumar"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Use the email on your GitHub account, or the private
`xxxxx+username@users.noreply.github.com` address from
**GitHub → Settings → Emails**.

## Step 3 — verify nothing dangerous is staged

```bash
git add -A
git status --short | head -40

# these MUST return nothing:
git status --porcelain | grep -E "secrets\.toml$|model\.pkl|ratings\.csv|movies\.csv"

# this MUST list 5 files (the app loads them at runtime):
git ls-files --others --cached app/bundle | sort
```

If `app/bundle` is empty, the deployed app will fail to start.

## Step 4 — commit and push

```bash
git commit -m "Add deep autoencoder, Streamlit app with TMDB, restructure into packages"
git push
```

If you also want the first commit re-attributed to you:

```bash
git commit --amend --reset-author --no-edit
git push --force-with-lease
```

## Step 5 — verify it works from a clean clone

This catches the single most common deployment failure — a file that exists on
your machine but was never committed:

```bash
cd /tmp
git clone https://github.com/Akshay113143/Movie-recommendation-System.git test-clone
cd test-clone
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If it opens and returns recommendations, Streamlit Cloud will work too.

## Step 6 — deploy

Follow `DEPLOYMENT.md`. Short version: share.streamlit.io → Create app →
repo `Akshay113143/Movie-recommendation-System`, branch `main`, main file
`streamlit_app.py` → Advanced settings → Secrets → paste
`TMDB_API_KEY = "your_key"` → Deploy.
