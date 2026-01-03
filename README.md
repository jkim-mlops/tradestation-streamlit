<!-- Local preview: Embedded looping video (may not work on GitHub) -->
<video src="assets/demo.mov" autoplay loop controls width="600"></video>

<!-- GitHub: Download or view the video -->
[Watch demo video](assets/demo.mov)

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

