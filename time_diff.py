#!/usr/bin/env python3
import socket
import time
import json
import statistics
import argparse
import datetime

def server_mode():
    """Run as server (on Mac)"""
    host = '0.0.0.0'  # Listen on all interfaces
    port = 12345
    
    print(f"Starting time difference server on port {port}")
    print("This will measure the time difference between your Mac and Raspberry Pi")
    print("Run the client mode on your Raspberry Pi")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((host, port))
    
    offsets = []
    round_trips = []
    
    try:
        for i in range(10):  # Run 10 measurements
            # Wait for message from client
            data, address = server_socket.recvfrom(1024)
            t2 = time.time()  # Server receive time
            
            client_data = json.loads(data.decode('utf-8'))
            t1 = client_data['t1']  # Client send time
            
            # Send response with timestamps
            response = {
                't1': t1,
                't2': t2,
                't3': time.time()  # Server send time
            }
            
            server_socket.sendto(json.dumps(response).encode('utf-8'), address)
            print(f"Completed measurement {i+1}/10")
            
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    finally:
        server_socket.close()
        print("Server closed")

def client_mode(server_ip):
    """Run as client (on Raspberry Pi)"""
    host = server_ip
    port = 12345
    
    print(f"Starting time difference client, connecting to {host}:{port}")
    print("This will measure the time difference between your Raspberry Pi and Mac")
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(5)  # 5 second timeout
    
    offsets = []
    round_trips = []
    
    try:
        for i in range(10):  # Run 10 measurements
            t1 = time.time()  # Client send time
            local_time = datetime.datetime.fromtimestamp(t1).strftime('%H:%M:%S.%f')
            
            # Send timestamp to server
            client_data = {
                't1': t1
            }
            client_socket.sendto(json.dumps(client_data).encode('utf-8'), (host, port))
            
            # Receive response
            try:
                data, server = client_socket.recvfrom(1024)
                t4 = time.time()  # Client receive time
                
                server_data = json.loads(data.decode('utf-8'))
                t1 = server_data['t1']  # Original client send time
                t2 = server_data['t2']  # Server receive time
                t3 = server_data['t3']  # Server send time
                
                # Calculate round-trip time and offset
                rtt = (t4 - t1) - (t3 - t2)
                offset = ((t2 - t1) + (t3 - t4)) / 2
                
                # Record measurements
                round_trips.append(rtt)
                offsets.append(offset)
                
                print(f"Measurement {i+1}/10:")
                print(f"  Local time: {local_time}")
                print(f"  Round-trip: {rtt*1000:.3f} ms")
                print(f"  Clock offset: {offset*1000:.3f} ms")
                print(f"  {'Your Pi is behind the Mac' if offset > 0 else 'Your Pi is ahead of the Mac'}")
                
            except socket.timeout:
                print(f"Timeout waiting for server response in measurement {i+1}")
                
            time.sleep(1)
            
        # Calculate statistics
        if offsets:
            avg_offset = statistics.mean(offsets)
            median_offset = statistics.median(offsets)
            min_offset = min(offsets)
            max_offset = max(offsets)
            
            avg_rtt = statistics.mean(round_trips) * 1000  # Convert to ms
            
            print("\n=== SUMMARY ===")
            print(f"Average network round-trip: {avg_rtt:.3f} ms")
            print(f"Clock difference statistics:")
            print(f"  Average offset: {avg_offset*1000:.3f} ms")
            print(f"  Median offset: {median_offset*1000:.3f} ms")
            print(f"  Min offset: {min_offset*1000:.3f} ms")
            print(f"  Max offset: {max_offset*1000:.3f} ms")
            print(f"  Jitter: {(max_offset-min_offset)*1000:.3f} ms")
            
            print("\nInterpretation:")
            if abs(median_offset) < 0.01:  # Less than 10ms
                print("✅ EXCELLENT: Clocks are very well synchronized!")
            elif abs(median_offset) < 0.05:  # Less than 50ms
                print("✓ GOOD: Clocks are well synchronized for most purposes")
            elif abs(median_offset) < 0.1:  # Less than 100ms
                print("⚠️ ACCEPTABLE: Clock difference may affect precise timing measurements")
            else:
                print("❌ POOR: Significant clock difference detected")
                
            direction = "behind" if median_offset > 0 else "ahead of"
            print(f"Your Raspberry Pi clock is {abs(median_offset*1000):.1f} ms {direction} your Mac")
            
            if median_offset < 0:
                print("\nThis explains your negative AoI values - your Pi's clock is ahead of your Mac")
                print("Recommend using the absolute value approach for AoI calculations")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        print("Client closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure time difference between two machines')
    parser.add_argument('mode', choices=['server', 'client'], help='Run as server (Mac) or client (Pi)')
    parser.add_argument('--server', '-s', help='Server IP address (needed for client mode)', default='')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        server_mode()
    elif args.mode == 'client':
        if not args.server:
            print("Error: Server IP address is required for client mode")
            print("Usage: python time_diff_checker.py client --server 192.168.x.x")
            exit(1)
        client_mode(args.server)
