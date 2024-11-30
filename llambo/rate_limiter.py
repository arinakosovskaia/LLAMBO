import time
import tiktoken


class RateLimiter:
    def __init__(self, max_tokens, time_frame, max_requests=450):
        # max number of tokens that can be used within time_frame
        self.max_tokens = max_tokens
        # max number of requests that can be made within time_frame
        self.max_requests = max_requests
        # time in seconds for which max_tokens is applicable
        self.time_frame = time_frame
        # keeps track of when tokens were used
        self.timestamps = []
        # keeps track of tokens used at each timestamp
        self.tokens_used = []
        # keeps track of the number of requests made
        self.request_count = 0

    def add_request(self, request_text=None, request_token_count=None, current_time=None, model='gpt-3.5-turbo'):
        current_time = time.time()

        if request_text is not None:
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            num_tokens = len(encoding.encode(request_text))
        elif request_token_count is not None:
            num_tokens = request_token_count
        else:
            raise ValueError('Either request_text or request_token_count must be specified.')


        while True:
            # Remove old requests outside the time frame
            while self.timestamps and self.timestamps[0] < current_time - self.time_frame:
                self.timestamps.pop(0)
                self.tokens_used.pop(0)
                self.request_count -= 1

            current_tokens = sum(self.tokens_used)
            print(f"[RateLimiter] Current tokens: {current_tokens}, Max tokens: {self.max_tokens}")
            print(f"[RateLimiter] Current requests: {self.request_count}, Max requests: {self.max_requests}")

            if self.request_count + 1 > self.max_requests or current_tokens + num_tokens > self.max_tokens:
                sleep_time = (self.timestamps[0] + self.time_frame) - current_time
                sleep_time = max(sleep_time, 0)  # Ensure non-negative sleep time
                print(f'[Rate Limiter] Sleeping for {sleep_time:.2f}s to avoid hitting the limit...')
                time.sleep(sleep_time)
                current_time = time.time()
            else:
                # Add new request
                self.timestamps.append(current_time)
                self.tokens_used.append(num_tokens)
                self.request_count += 1
                print(f"[RateLimiter] Request added. Tokens used: {sum(self.tokens_used)}, Requests made: {self.request_count}")
                break