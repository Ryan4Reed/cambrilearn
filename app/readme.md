# App for interacting with data

Simple Flask app that lets you view the predicted revenue distribution for a specific `account_id`.  

Users can enter an `account_id` via a web form to see:
- The **total expected revenue** for that account
- The **distribution per month** (18 months)

---

## Running the app
Please note that you will need to run the ml pipeline before being able to use this app.

Execute this command in your terminal
```bash
python app/app.py
```