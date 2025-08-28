
document.addEventListener('DOMContentLoaded', function() {
    const socket = io();
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatContainer = document.querySelector('.chat-container');

    socket.on('connect', () => {
        console.log('Conectado ao servidor! Aguardando histórico...');
    });

    socket.on('load_history', (data) => {
        const history = data.history;
        chatContainer.innerHTML = '';
        history.forEach(message => {
            if (message.role === 'user') {
                addUserMessage(message.parts.join(' '));
            } else if (message.role === 'model') {
                addAIMessage(message.parts.join(' '), false); 
            }
        });
        console.log('Histórico carregado.');
    });

    let currentAiBubble;
    socket.on('stream_start', () => {
        currentAiBubble = addAIMessage("", true);
    });

    socket.on('stream_chunk', (data) => {
        if (currentAiBubble) {
            const p = currentAiBubble.querySelector('p');
            p.innerHTML += data.data;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    });

    socket.on('stream_end', () => {
        if (currentAiBubble) {
            showTypingIndicator(currentAiBubble, false);
        }
        currentAiBubble = null;
    });

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            addUserMessage(message);
            userInput.value = '';
            socket.emit('new_message', { 'message': message });
        }
    });

    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'user-message';
        messageDiv.innerHTML = `<div class="user-bubble"><p>${text}</p></div>`;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function addAIMessage(text, isTyping) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'ai-message';
        messageDiv.innerHTML = `
            <div class="ai-bubble">
                <p>${text}</p>
            </div>
            <div class="typing-indicator">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>`;
        chatContainer.appendChild(messageDiv);
        showTypingIndicator(messageDiv, isTyping);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageDiv;
    }

    function showTypingIndicator(messageElement, show) {
        const indicator = messageElement.querySelector('.typing-indicator');
        if (indicator) {
            indicator.style.display = show ? 'flex' : 'none';
        }
    }
});