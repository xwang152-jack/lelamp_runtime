// Initialize LiveKit Room
let room = null;
const connectBtn = document.getElementById('connect-btn');
const serverUrlInput = document.getElementById('server-url');
const tokenInput = document.getElementById('token');
const connectionPanel = document.getElementById('connection-panel');
const roomPanel = document.getElementById('room-panel');
const videoContainer = document.getElementById('video-container');
const messagesDiv = document.getElementById('messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const disconnectBtn = document.getElementById('disconnect-btn');

async function connectToRoom() {
    const url = serverUrlInput.value.trim();
    const token = tokenInput.value.trim();

    if (!url || !token) {
        alert("请输入 Server URL 和 Token");
        return;
    }

    try {
        room = new LivekitClient.Room({
            adaptiveStream: true,
            dynacast: true,
        });

        // 监听事件
        room.on(LivekitClient.RoomEvent.TrackSubscribed, handleTrackSubscribed);
        room.on(LivekitClient.RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed);
        room.on(LivekitClient.RoomEvent.DataReceived, handleDataReceived);
        room.on(LivekitClient.RoomEvent.Disconnected, handleDisconnected);
        room.on(LivekitClient.RoomEvent.LocalTrackPublished, handleLocalTrackPublished);

        await room.connect(url, token);
        console.log('Connected to room', room.name);

        // UI 切换
        connectionPanel.classList.add('hidden');
        roomPanel.classList.remove('hidden');

        // 发布本地麦克风（如果需要双向语音）
        try {
            await room.localParticipant.enableCameraAndMicrophone();
        } catch (e) {
            console.warn("无法启用麦克风/摄像头:", e);
        }

    } catch (error) {
        console.error('Failed to connect:', error);
        alert('连接失败: ' + error.message);
    }
}

function handleTrackSubscribed(track, publication, participant) {
    if (track.kind === LivekitClient.Track.Kind.Video || track.kind === LivekitClient.Track.Kind.Audio) {
        // attach to a new element
        const element = track.attach();
        element.style.width = "100%";
        element.style.height = "100%";
        
        // Ensure video is visible
        if (track.kind === LivekitClient.Track.Kind.Video) {
            // Remove placeholder if exists
            const placeholder = videoContainer.querySelector('.placeholder');
            if (placeholder) placeholder.remove();
            
            videoContainer.appendChild(element);
        } else {
            // Audio elements don't need to be visible but must be in DOM
            document.body.appendChild(element);
        }
    }
}

function handleTrackUnsubscribed(track, publication, participant) {
    track.detach().forEach(element => element.remove());
}

function handleLocalTrackPublished(publication, participant) {
    // 可以在这里显示本地预览
}

function handleDataReceived(payload, participant) {
    const str = new TextDecoder().decode(payload);
    console.log('Received data:', str);
    
    // 简单的聊天显示
    try {
        const msg = JSON.parse(str);
        if (msg.type === 'chat') {
            appendMessage('agent', msg.content);
        }
    } catch (e) {
        // 如果不是 JSON，直接显示
        appendMessage('agent', str);
    }
}

function handleDisconnected() {
    room = null;
    connectionPanel.classList.remove('hidden');
    roomPanel.classList.add('hidden');
    videoContainer.innerHTML = '<div class="placeholder">等待摄像头画面...</div>';
}

async function sendChat(text) {
    if (!room) return;
    
    const encoder = new TextEncoder();
    const data = encoder.encode(JSON.stringify({
        type: 'chat',
        content: text
    }));

    // 发送数据到房间 (可靠传输)
    await room.localParticipant.publishData(data, LivekitClient.DataPacket_Kind.RELIABLE);
    appendMessage('user', text);
}

function appendMessage(sender, text) {
    const div = document.createElement('div');
    div.classList.add('msg', sender);
    div.textContent = text;
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Event Listeners
connectBtn.addEventListener('click', connectToRoom);
disconnectBtn.addEventListener('click', () => room && room.disconnect());
sendBtn.addEventListener('click', () => {
    const text = chatInput.value.trim();
    if (text) {
        sendChat(text);
        chatInput.value = '';
    }
});

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendBtn.click();
    }
});
