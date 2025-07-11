const sessionId = crypto.randomUUID();
const chatWindow = document.getElementById("chat-window");
const userInput  = document.getElementById("user-input");
const sendBtn    = document.getElementById("send-btn");

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", e => {
  if (e.key === "Enter") sendMessage();
});

function appendMessage(content, sender) {
  const msgEl = document.createElement("div");
  msgEl.classList.add("message", sender);
  msgEl.innerHTML = `<div class="bubble">${content}</div>`;
  chatWindow.appendChild(msgEl);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage(optionText = null) {
  let text;
  if (optionText) {
    text = optionText;
    appendMessage(text, "user");
  } else {
    text = userInput.value.trim();
    if (!text) return;
    appendMessage(text, "user");
    userInput.value = "";
  }

  const res = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ session_id: sessionId, message: text })
  });
  const data = await res.json();

  // 1) Structured JSON with follow-ups?
  if (data.structured && data.structured.question) {
    const { question, options } = data.structured;
    appendMessage(question, "bot");
    // render buttons
    const btnContainer = document.createElement("div");
    btnContainer.classList.add("options");
    options.forEach(opt => {
      const btn = document.createElement("button");
      btn.textContent = opt;
      btn.onclick = () => sendMessage(opt);
      btnContainer.appendChild(btn);
    });
    chatWindow.appendChild(btnContainer);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
  // 2) Final diagnosis JSON?
  else if (data.structured && data.structured.diagnosis) {
    const { diagnosis, explanation, resources } = data.structured;
    appendMessage(
      `<strong>Diagnosis:</strong> ${diagnosis}<br/>
       <em>${explanation}</em>`, "bot"
    );
    if (resources?.length) {
      const ul = document.createElement("ul");
      resources.forEach(url => {
        const li = document.createElement("li");
        li.innerHTML = `<a href="${url}" target="_blank">${url}</a>`;
        ul.appendChild(li);
      });
      chatWindow.appendChild(ul);
    }
  }
  // 3) Plain-text fallback
  else if (data.reply) {
    appendMessage(data.reply, "bot");
  }
}
