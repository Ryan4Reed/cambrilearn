from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv('data/revenue/revenue.csv')
df.set_index('account_id', inplace=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    account_id = ''

    if request.method == 'POST':
        account_id = request.form['account_id']
        if account_id in df.index:
            monthly_values = df.loc[account_id].tolist()
            total_revenue = sum(monthly_values)
            result = {
                'account_id': account_id,
                'total_revenue': total_revenue,
                'monthly_values': monthly_values
            }
        else:
            result = {
                'error': f"Account ID '{account_id}' not found."
            }

    return render_template('index.html', result=result, account_id=account_id)

if __name__ == '__main__':
    app.run(debug=True)
