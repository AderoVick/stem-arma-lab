# Deployment guide

TextPulse uses two separate hosting services:

- **GitHub Pages** hosts the static project website from `index.html`.
- **Streamlit Community Cloud** runs the Python dashboard from `streamlit_app.py`.

## 1. Publish the project website

Push all repository files to the `main` branch. In the GitHub repository, open:

```text
Settings → Pages
```

Use these settings:

```text
Source: Deploy from a branch
Branch: main
Folder: /(root)
```

Save the setting. The landing page will be available at:

```text
https://aderovick.github.io/stem-arma-lab/
```

The root `.nojekyll` file tells GitHub Pages to publish the static files directly.

## 2. Deploy the Streamlit dashboard

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app.
3. Select the repository `AderoVick/stem-arma-lab`.
4. Select branch `main`.
5. Set the main file path to `streamlit_app.py`.
6. Choose the app URL `stem-arma-lab` if it is available.
7. Deploy the app.

The expected address is:

```text
https://stem-arma-lab.streamlit.app/
```

No secrets or API keys are required.

## 3. Connect the landing page to the app

The dashboard URL is stored in:

```text
assets/site-config.js
```

It currently contains:

```javascript
window.TEXT_PULSE_CONFIG = {
  appUrl: "https://stem-arma-lab.streamlit.app/"
};
```

If Streamlit assigns a different address, replace only the `appUrl` value, commit the change, and push it to `main`.

## 4. Verify deployment

Check the following:

- The GitHub Pages website loads without showing the README as the homepage.
- The **Launch app** buttons open the Streamlit dashboard.
- The dashboard loads the built-in demonstration data.
- The GitHub Actions quality workflow passes.
- The website works on desktop and mobile screen sizes.

A hard refresh may be needed after publishing:

```text
Ctrl + F5
```
