import os
import asyncio
import base64
import json
import uuid
import urllib.request
from dotenv import load_dotenv

async def test_feishu_push():
    load_dotenv()
    
    app_id = os.getenv("FEISHU_APP_ID")
    app_secret = os.getenv("FEISHU_APP_SECRET")
    receive_id = os.getenv("FEISHU_RECEIVE_ID")
    receive_id_type = os.getenv("FEISHU_RECEIVE_ID_TYPE") or "open_id"
    
    print(f"Testing Feishu Push with:")
    print(f"APP_ID: {app_id}")
    print(f"RECEIVE_ID: {receive_id}")
    print(f"RECEIVE_ID_TYPE: {receive_id_type}")

    if not all([app_id, app_secret, receive_id]):
        print("Error: Missing environment variables.")
        return

    # Use a dummy image or a real one from assets
    image_path = "assets/images/1_lamp_3d.png"
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    else:
        # Create a tiny 1x1 black JPEG if assets not found
        image_bytes = base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAAAP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACE QUALITY")

    try:
        # 1. Get Token
        print("Step 1: Getting Tenant Access Token...")
        token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        token_payload = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8")
        token_req = urllib.request.Request(
            token_url, data=token_payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        
        token_resp_raw = urllib.request.urlopen(token_req, timeout=10).read()
        token_resp = json.loads(token_resp_raw)
        token = token_resp.get("tenant_access_token")
        if not token:
            print(f"Failed to get token: {token_resp}")
            return

        # 2. Upload Image
        print("Step 2: Uploading image...")
        upload_url = "https://open.feishu.cn/open-apis/im/v1/images"
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        
        body = []
        body.append(f"--{boundary}".encode("utf-8"))
        body.append(b'Content-Disposition: form-data; name="image_type"')
        body.append(b"")
        body.append(b"message")
        body.append(f"--{boundary}".encode("utf-8"))
        body.append(b'Content-Disposition: form-data; name="image"; filename="test.png"')
        body.append(b"Content-Type: image/png")
        body.append(b"")
        body.append(image_bytes)
        body.append(f"--{boundary}--".encode("utf-8"))
        
        payload = b"\r\n".join(body)
        upload_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        upload_req = urllib.request.Request(upload_url, data=payload, headers=upload_headers, method="POST")
        upload_resp = json.loads(urllib.request.urlopen(upload_req, timeout=15).read())
        
        image_key = upload_resp.get("data", {}).get("image_key")
        if not image_key:
            print(f"Failed to upload image: {upload_resp}")
            return
        print(f"Image uploaded, key: {image_key}")

        # 3. Send Message
        print("Step 3: Sending message...")
        msg_url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}"
        msg_payload = json.dumps({
            "receive_id": receive_id,
            "msg_type": "image",
            "content": json.dumps({"image_key": image_key})
        }).encode("utf-8")
        
        msg_req = urllib.request.Request(
            msg_url, data=msg_payload, 
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, 
            method="POST"
        )
        try:
            msg_resp_raw = urllib.request.urlopen(msg_req, timeout=10).read()
            msg_resp = json.loads(msg_resp_raw)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            print(f"HTTP Error {e.code}: {err_body}")
            return
        
        if msg_resp.get("code") == 0:
            print("Success! Test message sent to Feishu.")
        else:
            print(f"Failed to send message: {msg_resp}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_feishu_push())
