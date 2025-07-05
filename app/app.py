from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv("data/revenue/revenue.csv")
df.set_index("account_id", inplace=True)


MONTH_LABELS = [
    "July 2024",
    "August 2024",
    "September 2024",
    "October 2024",
    "November 2024",
    "December 2024",
    "January 2025",
    "February 2025",
    "March 2025",
    "April 2025",
    "May 2025",
    "June 2025",
    "July 2025",
    "August 2025",
    "September 2025",
    "October 2025",
    "November 2025",
    "December 2025",
]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    account_id = ""

    if request.method == "POST":
        account_id = request.form.get("account_id", "").strip()

        if account_id in df.index:
            monthly_values = df.loc[account_id].tolist()
            total_revenue = sum(monthly_values)
            monthly_distribution = [
                (label.split(" ")[0], label.split(" ")[1], revenue)
                for label, revenue in zip(MONTH_LABELS, monthly_values)
            ]

            result = {
                "account_id": account_id,
                "total_revenue": total_revenue,
                "monthly_distribution": monthly_distribution,
            }
        else:
            result = {"error": f"Account ID '{account_id}' not found."}

    return render_template("index.html", result=result, account_id=account_id)


if __name__ == "__main__":
    app.run(debug=True)
