import os
import sqlite3
from market import Market
from utils import * 

def main():
    """
    Run this file to run the main function which starts the macro ABM simulation.
    """
    # load model parameters
    params = load_parameters('parameters.yml')
    # create market object
    market = Market(params)
    # database path
    parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    database_path = f"{parent_path}\\data"
    # connect to database
    conn = sqlite3.connect(f"{database_path}\\{params['database_name']}.db")
    cur = conn.cursor()
    # run simulation
    market.run_simulation(cur)
    # close database connection 
    cur.close()
    conn.commit()
    conn.close()

# run main function
if __name__ == '__main__':
    main()
