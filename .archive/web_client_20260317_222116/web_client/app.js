// ==================== Global State ====================
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

// ==================== Toast 通知系统 ====================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };

    toast.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ==================== Tab 切换 ====================
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.dataset.tab;

        // 更新按钮状态
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // 更新面板显示
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        document.getElementById(`tab-${targetTab}`).classList.add('active');
    });
});

// ==================== LiveKit 连接 ====================
async function connectToRoom() {
    const url = serverUrlInput.value.trim();
    const token = tokenInput.value.trim();

    if (!url || !token) {
        showToast('请输入 Server URL 和 Token', 'error');
        return;
    }

    try {
        showToast('正在连接...', 'info');

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

        showToast('连接成功！', 'success');

        // 发布本地麦克风（如果需要双向语音）
        try {
            await room.localParticipant.enableCameraAndMicrophone();
        } catch (e) {
            console.warn("无法启用麦克风/摄像头:", e);
        }

    } catch (error) {
        console.error('Failed to connect:', error);
        showToast('连接失败: ' + error.message, 'error');
    }
}

function handleTrackSubscribed(track, publication, participant) {
    if (track.kind === LivekitClient.Track.Kind.Video || track.kind === LivekitClient.Track.Kind.Audio) {
        const element = track.attach();
        element.style.width = "100%";
        element.style.height = "100%";

        if (track.kind === LivekitClient.Track.Kind.Video) {
            // 移除占位符
            const placeholder = videoContainer.querySelector('.placeholder');
            if (placeholder) placeholder.remove();

            videoContainer.appendChild(element);

            // 更新隐私指示器
            updateCameraStatus(true);
        } else {
            // 音频元素添加到 DOM 但不可见
            document.body.appendChild(element);
        }
    }
}

function handleTrackUnsubscribed(track, publication, participant) {
    track.detach().forEach(element => element.remove());

    if (track.kind === LivekitClient.Track.Kind.Video) {
        updateCameraStatus(false);
    }
}

function handleLocalTrackPublished(publication, participant) {
    console.log('Local track published:', publication);
}

function handleDataReceived(payload, participant) {
    const str = new TextDecoder().decode(payload);
    console.log('Received data:', str);

    try {
        const msg = JSON.parse(str);

        if (msg.type === 'chat') {
            appendMessage('agent', msg.content);
        } else if (msg.type === 'vision_result') {
            // 展示视觉结果
            showVisionResult(msg.content, msg.image_base64);
        } else if (msg.type === 'camera_status') {
            // 更新摄像头状态
            updateCameraStatus(msg.active);
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
    videoContainer.innerHTML = '<div class="placeholder"><div class="placeholder-icon">📹</div><p>等待摄像头画面...</p></div>';
    showToast('已断开连接', 'info');
}

// ==================== 消息发送 ====================
async function sendChat(text) {
    if (!room) {
        showToast('请先连接设备', 'error');
        return;
    }

    const encoder = new TextEncoder();
    const data = encoder.encode(JSON.stringify({
        type: 'chat',
        content: text
    }));

    try {
        await room.localParticipant.publishData(data, LivekitClient.DataPacket_Kind.RELIABLE);
        appendMessage('user', text);
    } catch (error) {
        console.error('Failed to send message:', error);
        showToast('发送失败', 'error');
    }
}

async function sendCommand(action, params = {}) {
    if (!room) {
        showToast('请先连接设备', 'error');
        return;
    }

    const command = {
        type: 'command',
        action: action,
        params: params
    };

    const encoder = new TextEncoder();
    const data = encoder.encode(JSON.stringify(command));

    try {
        await room.localParticipant.publishData(data, LivekitClient.DataPacket_Kind.RELIABLE);
    } catch (error) {
        console.error('Failed to send command:', error);
        showToast('指令发送失败', 'error');
    }
}

function appendMessage(sender, text) {
    const div = document.createElement('div');
    div.classList.add('msg', sender);
    div.textContent = text;
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function clearMessages() {
    messagesDiv.innerHTML = '';
    showToast('消息已清空', 'info');
}

// ==================== 视觉功能 ====================
async function captureAndAsk() {
    showToast('📸 正在拍照识别...', 'info');
    await sendChat('这是什么');
}

async function checkHomework() {
    showToast('📚 正在检查作业，请稍候...', 'info');
    await sendChat('帮我检查作业');
}

async function sendToFeishu() {
    showToast('✈️ 正在推送到飞书...', 'info');
    await sendChat('拍照发送到飞书');
}

function showVisionResult(content, imageBase64 = null) {
    const resultDiv = document.getElementById('vision-result');
    const responseText = document.getElementById('vision-response');
    const capturedImage = document.getElementById('captured-image');

    responseText.textContent = content;

    if (imageBase64) {
        capturedImage.src = `data:image/jpeg;base64,${imageBase64}`;
        capturedImage.style.display = 'block';
    } else {
        capturedImage.style.display = 'none';
    }

    resultDiv.classList.remove('hidden');
}

function closeVisionResult() {
    document.getElementById('vision-result').classList.add('hidden');
}

// ==================== 动作控制 ====================
async function playAnimation(animName) {
    showToast(`🎭 正在播放动画: ${animName}`, 'info');
    await sendCommand('play_recording', { recording_name: animName });
}

// ==================== 灯光控制 ====================
function setCustomColor() {
    const colorInput = document.getElementById('light-color');
    const hex = colorInput.value;

    // 转换 HEX 到 RGB
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);

    setRgbColor(r, g, b);
}

async function setRgbColor(r, g, b) {
    showToast(`💡 设置灯光颜色: RGB(${r}, ${g}, ${b})`, 'success');
    await sendCommand('set_rgb_solid', { r, g, b });
}

async function setRgbEffect(effectName) {
    const effectNames = {
        breathing: '呼吸灯',
        rainbow: '彩虹',
        wave: '波浪',
        fire: '火焰',
        fireworks: '烟花',
        starry: '星空'
    };

    showToast(`💡 启动灯效: ${effectNames[effectName]}`, 'success');
    await sendCommand(`rgb_effect_${effectName}`, {});
}

// ==================== 隐私控制 ====================
function updateCameraStatus(active) {
    const led = document.getElementById('camera-led');
    const statusText = document.getElementById('camera-status-text');

    if (active) {
        led.classList.add('active');
        statusText.textContent = '摄像头已激活';
        statusText.style.color = 'var(--danger)';
    } else {
        led.classList.remove('active');
        statusText.textContent = '摄像头已关闭';
        statusText.style.color = 'var(--text-muted)';
    }
}

// ==================== Event Listeners ====================
connectBtn.addEventListener('click', connectToRoom);

disconnectBtn.addEventListener('click', () => {
    if (room) {
        room.disconnect();
    }
});

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

// ==================== 页面加载完成 ====================
window.addEventListener('load', () => {
    console.log('LeLamp Web Client v2.0 加载完成');
});
