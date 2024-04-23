import asyncio
import websockets
import time

async def send_file(url, file_path, client_id):
    async with websockets.connect(url) as websocket:
        response_times = []
        inference_times = []
        for i in range(10):  # Loop to send the file 100 times
            with open(file_path, "rb") as file:
                data = file.read()
                network_send_start = time.time()
                await websocket.send(data)  # Send the entire file data as binary
                network_send_end = time.time()
                network_send_time = network_send_end - network_send_start
                
                response_start = time.time()
                response = await websocket.recv()  # Wait for a response from the server
                response_end = time.time()
                inference_time = response_end - response_start
                
                response_times.append(network_send_time + inference_time)
                inference_times.append(inference_time)
                print(f"Client {client_id}, transmission {i+1}: Network Send Time={network_send_time:.4f} seconds, Inference Time={inference_time:.4f} seconds")
        
        # Calculate the average response time
        average_response_time = sum(response_times) / len(response_times)
        average_inference_time = sum(inference_times) / len(inference_times)
        print(f"Client {client_id} average response time: {average_response_time:.4f} seconds")
        print(f"Client {client_id} average inference time: {average_inference_time:.4f} seconds")

async def main(url, file_path, num_clients):
    tasks = [send_file(url, file_path, i) for i in range(num_clients)]
    await asyncio.gather(*tasks)  # Run all client tasks concurrently

# Replace 'localhost' and '8000' with your FastAPI server's host and port
url = "ws://0.0.0.0:9001/ws_persistent_transcribe"
file_path = "audios/10sec.wav"  # Path of the file you want to send
num_clients = 100  # Number of concurrent clients

# Run the stress test with multiple clients
asyncio.run(main(url, file_path, num_clients))
