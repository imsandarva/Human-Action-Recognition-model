import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:har_app/services/api_service.dart';
import 'package:har_app/services/sensor_service.dart';

class TestDataScreen extends StatefulWidget {
  const TestDataScreen({super.key});
  @override
  State<TestDataScreen> createState() => _TestDataScreenState();
}

class _TestDataScreenState extends State<TestDataScreen> {
  final SensorService _sensor = SensorService();
  final ApiService _api = ApiService();
  final List<String> _logs = [];
  bool _active = false, _showInfo = true;
  String _activity = 'WAITING', _status = 'Disconnected';
  double _conf = 0.0;

  void _toggle() {
    setState(() => _active = !_active);
    _active ? _sensor.startSampling(_onData) : _sensor.stopSampling();
    setState(() => _status = _active ? 'Live' : 'Stopped');
  }

  Future<void> _onData(List<List<double>> d) async {
    final res = await _api.predictActivity(d);
    if (!mounted) return;
    setState(() {
      _status = res != null ? 'Connected' : 'Sync Error (Retrying)';
      if (res != null) {
        _activity = res['label'] ?? 'UNKNOWN';
        _conf = (res['confidence'] ?? 0.0) * 100;
      }
      _logs.insert(0, '${DateTime.now().second}s: ${res != null ? (res['label'] ?? "Success") : "Failed"}');
      if (_logs.length > 10) _logs.removeLast();
    });
  }

  @override
  void dispose() {
    _sensor.stopSampling();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: const Color(0xFFF1F5F9),
    body: Stack(
      children: [
        SafeArea(child: Padding(padding: const EdgeInsets.all(28), child: Column(children: [
          _buildTopScope(context), const Spacer(), _buildMainStage(), const Spacer(), _buildLogs(), const SizedBox(height: 26), _buildAction(),
        ]))),
        if (_showInfo) _buildInstructionOverlay(),
      ],
    ),
  );

  Widget _buildTopScope(BuildContext context) => Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
    Row(children: [
      IconButton(onPressed: () => Navigator.of(context).pop(), icon: const Icon(Icons.arrow_back, color: Color(0xFF0F172A))),
      Text('Test Data', style: GoogleFonts.outfit(fontWeight: FontWeight.w900, color: const Color(0xFF0F172A))),
    ]),
    Container(padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4), decoration: BoxDecoration(color: _active ? Colors.green.withOpacity(0.1) : Colors.black12, borderRadius: BorderRadius.circular(20)), child: Text(_status, style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: _active ? Colors.green : Colors.black54))),
  ]);

  Widget _buildMainStage() => Column(children: [
    Text(_activity, style: GoogleFonts.outfit(fontSize: 46, fontWeight: FontWeight.w900, color: const Color(0xFF0F172A), letterSpacing: -1)),
    Text('${_conf.toStringAsFixed(1)}% CONFIDENCE', style: GoogleFonts.outfit(fontSize: 14, fontWeight: FontWeight.w600, color: const Color(0xFF6366F1))),
  ]);

  Widget _buildLogs() => Container(height: 110, width: double.infinity, padding: const EdgeInsets.all(16), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(18)), 
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('SERVER LOGS', style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Colors.black38)),
      Expanded(child: ListView.builder(itemCount: _logs.length, itemBuilder: (c, i) => Text(_logs[i], style: GoogleFonts.jetBrainsMono(fontSize: 11, color: Colors.black87)))),
    ]));

  Widget _buildAction() => GestureDetector(onTap: _toggle, child: Container(height: 76, width: double.infinity, decoration: BoxDecoration(borderRadius: BorderRadius.circular(24), gradient: LinearGradient(colors: _active ? [Colors.redAccent, Colors.red] : [const Color(0xFF6366F1), const Color(0xFF4F46E5)]), boxShadow: [BoxShadow(color: (_active ? Colors.red : const Color(0xFF6366F1)).withOpacity(0.3), blurRadius: 20, offset: const Offset(0, 10))]), child: Center(child: Text(_active ? 'STOP SESSION' : 'START TEST DATA', style: GoogleFonts.outfit(fontSize: 17, fontWeight: FontWeight.w800, color: Colors.white)))));

  Widget _buildInstructionOverlay() => Container(color: Colors.black.withOpacity(0.9), width: double.infinity, child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
    const Icon(Icons.phonelink_ring, color: Colors.white, size: 80), const SizedBox(height: 24),
    Text('POCKET PORTRAIT', style: GoogleFonts.outfit(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)),
    const Padding(padding: EdgeInsets.all(32), child: Text('Place the phone in your pocket with the screen facing your body for ideal results.', textAlign: TextAlign.center, style: TextStyle(color: Colors.white70, fontSize: 16))),
    TextButton(onPressed: () => setState(() => _showInfo = false), child: const Text('GOT IT', style: TextStyle(color: Color(0xFF6366F1), fontWeight: FontWeight.bold, fontSize: 18))),
  ]));
}
