STOP: This is not to be put into production under any circumstances
This is just for demonstrating what is possible with Document intelligence. 

to run the demo, follow these commands 

cd [root]

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
Copy-Item exmple.env .env

python app.py