import 'dart:async';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:sensors_plus/sensors_plus.dart';

class AccSample {
  final int tsMs;
  final double ax, ay, az;
  const AccSample(this.tsMs, this.ax, this.ay, this.az);
}

class LabeledWindow {
  final int id;
  final String label;
  final int startTsMs;
  final double avgRate;
  final List<AccSample> samples;
  const LabeledWindow({required this.id, required this.label, required this.startTsMs, required this.avgRate, required this.samples});
}

class DataCollectionResult {
  final String label;
  final int samplingRateRequested;
  final Duration duration;
  final List<LabeledWindow> windows;
  final bool lowRateDetected;
  final String orientationHint;
  const DataCollectionResult({required this.label, required this.samplingRateRequested, required this.duration, required this.windows, required this.lowRateDetected, required this.orientationHint});
  double get avgRateOverAll => windows.isEmpty ? 0.0 : windows.map((w) => w.avgRate).reduce((a, b) => a + b) / windows.length;
}

class DataCollectionService {
  DataCollectionService({this.samplingRateHz = 50, this.windowSize = 100, this.lowRateThresholdHz = 40});
  final int samplingRateHz;
  final int windowSize;
  final int lowRateThresholdHz;
  StreamSubscription<AccelerometerEvent>? _sub;
  final List<AccSample> _allSamples = [];
  final List<AccSample> _currentWindow = [];
  final List<LabeledWindow> _windows = [];
  final List<int> _recentTsMs = [];
  String? _label;
  int? _lowRateStartMs;
  bool _lowRateDetected = false;

  bool get isRecording => _sub != null;

  void start(String label) {
    if (isRecording) return;
    _reset();
    _label = label;
    _sub = accelerometerEventStream(samplingPeriod: Duration(milliseconds: (1000 / samplingRateHz).round())).listen(_onEvent);
  }

  DataCollectionResult stop({required Duration duration, required String orientationHint}) {
    _sub?.cancel();
    _sub = null;
    if (_currentWindow.length < windowSize) _currentWindow.clear(); // Discard partial window per spec.
    final result = DataCollectionResult(label: _label ?? 'UNKNOWN', samplingRateRequested: samplingRateHz, duration: duration, windows: List.unmodifiable(_windows), lowRateDetected: _lowRateDetected, orientationHint: orientationHint);
    _label = null;
    return result;
  }

  void dispose() {
    _sub?.cancel();
    _sub = null;
    _reset();
  }

  void _onEvent(AccelerometerEvent e) {
    final ts = DateTime.now().millisecondsSinceEpoch;
    final sample = AccSample(ts, e.x, e.y, e.z);
    _allSamples.add(sample); // Keep all raw samples in memory until saved.
    _currentWindow.add(sample);
    _trackRate(ts);
    if (_currentWindow.length == windowSize) _commitWindow();
  }

  void _commitWindow() {
    final label = _label;
    if (label == null) return;
    final samples = List<AccSample>.from(_currentWindow);
    _currentWindow.clear();
    final start = samples.first.tsMs;
    final end = samples.last.tsMs;
    final durationMs = max(1, end - start);
    final avgRate = samples.length / (durationMs / 1000);
    _windows.add(LabeledWindow(id: _windows.length, label: label, startTsMs: start, avgRate: avgRate, samples: List.unmodifiable(samples)));
  }

  void _trackRate(int tsMs) {
    _recentTsMs.add(tsMs);
    while (_recentTsMs.isNotEmpty && tsMs - _recentTsMs.first > 1000) { _recentTsMs.removeAt(0); }
    if (_recentTsMs.length < lowRateThresholdHz) {
      _lowRateStartMs ??= tsMs;
      if (!_lowRateDetected && tsMs - _lowRateStartMs! > 1000) {
        _lowRateDetected = true;
        debugPrint('Warning: sample rate below ${lowRateThresholdHz}Hz for over 1s.');
      }
    } else {
      _lowRateStartMs = null;
    }
  }

  void _reset() {
    _allSamples.clear();
    _currentWindow.clear();
    _windows.clear();
    _recentTsMs.clear();
    _lowRateStartMs = null;
    _lowRateDetected = false;
  }
}
