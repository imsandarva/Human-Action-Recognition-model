import 'dart:async';
import 'package:sensors_plus/sensors_plus.dart';

class SensorService {
  final List<List<double>> _buffer = [];
  StreamSubscription? _accSub;
  Timer? _timer;

  void startSampling(Function(List<List<double>>) onWindowReady) {
    // 50Hz Sampling (20ms)
    _accSub = accelerometerEventStream(samplingPeriod: const Duration(milliseconds: 20)).listen((e) {
      _buffer.add([e.x, e.y, e.z]);
      if (_buffer.length > 200) _buffer.removeAt(0); // Extra safety buffer
    });

    // Send every 1 second (50% overlap of 100 samples @ 50Hz)
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (_buffer.length >= 100) {
        final window = _buffer.sublist(_buffer.length - 100);
        onWindowReady(window);
      }
    });
  }

  void stopSampling() {
    _accSub?.cancel();
    _timer?.cancel();
    _buffer.clear();
  }
}
