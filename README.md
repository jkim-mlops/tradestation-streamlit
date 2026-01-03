# tradestation-streamlit
Streamlit app backed by TradeStation used to visualize stock charts.

## Usage

1. Create a `.env` file with the TradeStation `client id`, `client secret`, and `refresh token` environment variables set.

    ```bash
    TS_AUTH_CLIENT_ID=<your client id>
    TS_AUTH_CLIENT_SECRET=<your client secret>
    TS_AUTH_REFRESH_TOKEN=<your refresh token>
    ```

1. Run the app locally.

    ```bash
    streamlit run main.py
    ```

https://github.com/user-attachments/assets/4c7303e6-89f9-4db5-adc8-765ea04b259e