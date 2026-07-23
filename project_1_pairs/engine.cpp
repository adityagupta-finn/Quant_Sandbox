#include <iostream>
#include <string>
#include <sqlite3.h>

using namespace std;

int main() {
    sqlite3* db;
    // 1. Open the pipeline to the database
    int exit = sqlite3_open("market_data.db", &db);
    
    if (exit) {
        cerr << "CRITICAL ERROR: Cannot open database: " << sqlite3_errmsg(db) << endl;
        return -1;
    } else {
        cout << "System Online: Connected to Alpha Database." << endl;
    }

    // 2. Query the absolute latest data point (Limit 1)
    string sql = "SELECT Date, Z_Score FROM pairs_data ORDER BY Date DESC LIMIT 1;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, NULL) == SQLITE_OK) {
        
        // 3. Read the data row
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            string date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            double z_score = sqlite3_column_double(stmt, 1);

            cout << "\n--- LIVE MARKET SCAN ---" << endl;
            cout << "Date: " << date << " | Z-Score: " << z_score << endl;

            // 4. The Algorithmic Trade Execution Logic
            if (z_score >= 2.0) {
                cout << "SIGNAL: EXTREME DIVERGENCE (OVERBOUGHT)." << endl;
                cout << "ACTION: EXECUTING SPREAD -> SHORT AAPL, LONG MSFT." << endl;
            } else if (z_score <= -2.0) {
                cout << "SIGNAL: EXTREME DIVERGENCE (OVERSOLD)." << endl;
                cout << "ACTION: EXECUTING SPREAD -> LONG AAPL, SHORT MSFT." << endl;
            } else {
                cout << "SIGNAL: MEAN REVERTED. NO STATISTICAL EDGE." << endl;
                cout << "ACTION: DO NOTHING. HOLDING CASH." << endl;
            }
            cout << "------------------------\n" << endl;
        }
    } else {
        cerr << "Database Query Failed." << endl;
    }

    // 5. Safely close the database connection
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    
    return 0;
}