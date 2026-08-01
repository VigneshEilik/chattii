class ChatiiClient {
    /**
     * JS Client wrapper accessing Chatii REST/WebSocket APIs.
     * @param {string} baseUrl
     * @param {string} apiKey
     */
    constructor(baseUrl = "http://localhost:8000/v1", apiKey = null) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
    }

    async getHeaders() {
        const headers = { "Content-Type": "application/json" };
        if (this.apiKey) {
            headers["Authorization"] = `Bearer ${this.apiKey}`;
        }
        return headers;
    }

    async chatCompletion(model, messages, options = {}) {
        const url = `${this.baseUrl}/chat/completions`;
        const payload = {
            model,
            messages,
            temperature: options.temperature ?? 0.7,
            top_p: options.top_p ?? 0.9,
            max_tokens: options.max_tokens ?? 128,
            stream: options.stream ?? false
        };

        const response = await fetch(url, {
            method: "POST",
            headers: await this.getHeaders(),
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API Error ${response.status}: ${await response.text()}`);
        }

        return options.stream ? response.body : response.json();
    }
}

if (typeof module !== "undefined") {
    module.exports = { ChatiiClient };
}
