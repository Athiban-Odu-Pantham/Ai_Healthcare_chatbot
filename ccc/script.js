const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function addMessage(text, sender) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('chat-message', sender);
  msgDiv.innerText = text;
  chatWindow.appendChild(msgDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

window.onload = () => {
  addMessage("Hello! I'm your healthcare assistant. How can I help you today?", "bot");
};

function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";

  fetch("http://127.0.0.1:5000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  })
    .then(response => response.json())
    .then(data => {
      addMessage(data.bot, "bot");
    })
    .catch(err => {
      console.error(err);
      addMessage("Error connecting to server.", "bot");
    });
}

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', function(e) {
  if (e.key === 'Enter') {
    sendMessage();
  }
});
