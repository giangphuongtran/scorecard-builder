"""Static copy for the publishable Streamlit workbench landing section.

Written in plain English on purpose: this workbench is often the first thing
a non-technical reviewer (recruiter, hiring manager, product person) opens,
so jargon is explained inline rather than assumed.
"""

GITHUB_URL = "https://github.com/<username>/credit-scoring"

TITLE = "Credit Scoring: From Risk Score to Lending Decision"

TAGLINE = (
    "A complete walkthrough of how a bank could decide who gets a loan: turn raw "
    "application and payment data into a risk score, convert that score into a "
    "probability of default, and use that probability to make a profitable, "
    "explainable approve/decline decision — for two loan types, installment (INS) "
    "and cash/card (CSS)."
)

# GitHub mark as inline SVG so the badge renders with a real icon, with no
# external image request needed (works offline / behind restrictive networks).
_GITHUB_SVG = (
    '<svg viewBox="0 0 16 16" width="18" height="18" fill="#ffffff" '
    'style="vertical-align:-3px;margin-right:6px;">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
    '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53 '
    '.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
    '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 '
    '1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 '
    '3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>'
    "</svg>"
)


def github_badge(url: str = GITHUB_URL, text: str = "View source on GitHub") -> str:
    """Return an HTML button-style badge with a real GitHub icon."""
    return (
        f'<a href="{url}" target="_blank" style="text-decoration:none;">'
        f'<div style="display:inline-flex;align-items:center;background:#181717;'
        f'color:#ffffff;padding:6px 14px;border-radius:6px;font-size:14px;'
        f'font-weight:500;">{_GITHUB_SVG}{text}</div></a>'
    )


# Each phase: (short title, one-line plain-English explanation)
PIPELINE_PHASES = [
    (
        "1. Explore the data",
        "Look at who applied, who defaulted, and how the two loan types differ.",
    ),
    (
        "2. Build behavior features",
        "Summarize each customer's last 3/6/9/12 months of payment history into "
        "simple numbers a model can use (e.g. 'how often were they late?').",
    ),
    (
        "3. Assemble a monthly dataset",
        "Recreate a month-by-month snapshot of the portfolio, so testing looks "
        "like real bank operations instead of one random shuffle of the data.",
    ),
    (
        "4. Build the scorecard",
        "Group each variable into readable risk bands, then fit a transparent "
        "statistical model that turns those bands into a risk score.",
    ),
    (
        "5. Design the approval policy",
        "Convert the risk score into a probability of default, then choose "
        "cutoffs that balance approving enough loans against losing money on "
        "defaults.",
    ),
    (
        "6. Report the results",
        "This dashboard: how good the model is, what the policy decides, and a "
        "tool to test one application at a time.",
    ),
    (
        "7. Serve it live",
        "A small API other systems could call to get a live score and "
        "decision, plus experiment tracking.",
    ),
]

MEASURED_OUTCOMES = [
    (
        "How well the model tells risky and safe applicants apart (INS)",
        "75.5%",
        "Gini score — higher is better; 100% would be a perfect ranking.",
    ),
    (
        "How well the model tells risky and safe applicants apart (CSS)",
        "52.8%",
        "Gini score for the second, harder-to-predict loan type.",
    ),
    (
        "Profit if this policy had been used historically",
        "≈ 965,000 PLN",
        "Estimated by replaying the policy on 1975–1987 data — not a live result, see note below.",
    ),
]

DATA_DISCLAIMER = (
    "The data behind this project is a licensed academic dataset and is not "
    "included in the code repository. Every profit figure shown here is an "
    "**estimate computed on historical data, after the fact** — think of it as "
    "\"what would have happened,\" not proof of an ongoing, live result."
)

STACK = "Python · pandas · scikit-learn · statsmodels · Kedro · Streamlit · FastAPI · MLflow"

# ---------------------------------------------------------------------------
# Plain-English glossary for technical terms used throughout the workbench.
# Shown as an expander on the landing page and used for tooltips elsewhere.
# ---------------------------------------------------------------------------
GLOSSARY = {
    "PD (Probability of Default)": "The model's estimate of how likely an applicant is to fail to repay, shown as a percentage. Lower is safer.",
    "Score": "A points-style number the model gives each applicant. It ranks risk but isn't itself a percentage — PD is the easier-to-read, converted version.",
    "Gini": "A single number (0-100%) that measures how well the model tells risky and safe applicants apart. Higher means better separation.",
    "WOE (Weight of Evidence)": "A technique that groups each variable into a few readable bands (e.g. income: low / medium / high) before modeling, so the scorecard stays easy to explain.",
    "Calibration": "The step that converts the raw score into an actual probability, so \"5%\" really does mean roughly a 1-in-20 chance of default.",
    "Cutoff / threshold": "The risk level at which the policy switches from approving to declining an application.",
    "Mid-band": "A grey-zone risk range where the applicant is neither a clear approve nor a clear decline, so extra checks decide the outcome.",
    "Acceptance rate": "The share of applications that get approved under the policy.",
    "Bad rate": "Among approved loans, the share that actually end up defaulting.",
    "As-if / offline profit": "Profit calculated by replaying the policy over historical data. It shows what the policy would have earned, not a live, currently-running result.",
    "Stability": "Whether a variable behaves consistently over time. Unstable variables are excluded so the scorecard doesn't lean on something that stops working.",
}

# Note: cutoff labels/help text now live at the source in
# credit_scoring.scorecard.reports.CUTOFF_DISPLAY (kept in plain English there
# so every consumer of that dict benefits, not just this workbench).

DECISION_PLAIN = {
    "A": ("Approved", "This application meets the policy's risk requirements."),
    "D": ("Declined", "This application did not meet the policy's risk requirements."),
    "N": ("No decision", "The policy could not reach a clear outcome for this input."),
}