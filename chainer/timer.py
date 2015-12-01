from cupy import cuda

class Timer(object):
    def __init__(self):
        self.initted = False
        self.running = False
        self.has_run_at_least_once = False

        self._init()

    def __del__(self):
        self.start_gpu = None
        self.stop_gpu = None
        self.stream = None

    def start(self):
        if not self.running:
            self.start_gpu.record(self.stream)
            self.running = True
            self.has_run_at_least_once = True

    def stop(self):
        if self.running:
            self.stop_gpu.record(self.stream)
            self.stop_gpu.synchronize()
            self.running = False

    def milliseconds(self):
        if not self.has_run_at_least_once:
            print('Timer has never been run before reading time.')
            return 0

        if self.running:
            self.stop()

        self.elapsed_milliseconds = cuda.stream.get_elapsed_time(self.start_gpu, self.stop_gpu)
        return self.elapsed_milliseconds

    def microseconds(self):
        if not self.has_run_at_least_once:
            print('Timer has never been run before reading time.')
            return 0

        if self.running:
            self.stop()
            
        self.elapsed_milliseconds = cuda.stream.get_elapsed_time(self.start_gpu, self.stop_gpu) * 1000
        return self.elapsed_microseconds

    def seconds(self):
        self.milliseconds() / 1000.0

    def _init(self):
        if not self.initted:
            self.start_gpu = cuda.Event()
            self.stop_gpu = cuda.Event()
            self.stream = cuda.Stream()
        self.initted = True

