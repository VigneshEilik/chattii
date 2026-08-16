document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chatMessages");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const themeToggle = document.getElementById("themeToggle");
  const tempSlider = document.getElementById("tempSlider");
  const tempVal = document.getElementById("tempVal");
  const modelSelect = document.getElementById("modelSelect");

  let currentConversationId = null;

  // Theme toggle
  themeToggle.addEventListener("click", () => {
    const html = document.documentElement;
    const currentTheme = html.getAttribute("data-theme");
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", nextTheme);
    themeToggle.textContent = nextTheme === "dark" ? "🌙" : "☀️";
  });

  tempSlider.addEventListener("input", (e) => {
    tempVal.textContent = e.target.value;
  });

  // Append Message
  function appendMessage(role, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;
    
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "U" : "C";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  // Send Message
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = "";
    appendMessage("user", text);

    const assistantBubble = appendMessage("assistant", "Thinking...");

    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer chatii-admin-root-key"
        },
        body: JSON.stringify({
          model: modelSelect.value,
          messages: [{ role: "user", content: text }],
          temperature: parseFloat(tempSlider.value),
          max_tokens: 256
        })
      });

      if (!response.ok) {
        assistantBubble.textContent = "Error: Failed to reach Chatii inference server.";
        return;
      }

      const data = await response.json();
      if (data.choices && data.choices.length > 0) {
        assistantBubble.textContent = data.choices[0].message.content;
      } else {
        assistantBubble.textContent = "Received empty response.";
      }
    } catch (err) {
      assistantBubble.textContent = "Network error: " + err.message;
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  newChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = "";
    appendMessage("assistant", "New conversation started. How can I help you?");
  });
});
