import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:har_app/services/api_service.dart';
import 'package:har_app/services/data_collection_service.dart';
import 'package:har_app/services/dataset_export_service.dart';
import 'package:share_plus/share_plus.dart';

enum CollectionPhase { idle, chooseLabel, countdown, recording, complete }

class DataCollectionScreen extends StatefulWidget {
  const DataCollectionScreen({super.key});
  @override
  State<DataCollectionScreen> createState() => _DataCollectionScreenState();
}

class _DataCollectionScreenState extends State<DataCollectionScreen> {
  final DataCollectionService _collector = DataCollectionService();
  final DatasetExportService _exporter = DatasetExportService();
  final ApiService _api = ApiService();
  final Duration _recordDuration = const Duration(minutes: 5);
  final int _countdownSeconds = 5;
  CollectionPhase _phase = CollectionPhase.idle;
  String _label = '';
  int _countdown = 5, _elapsedSeconds = 0;
  Timer? _countdownTimer, _elapsedTimer, _autoStopTimer;
  DataCollectionResult? _result;
  CollectionSaveResult? _saved;
  bool _busy = false;

  @override
  void dispose() {
    _countdownTimer?.cancel();
    _elapsedTimer?.cancel();
    _autoStopTimer?.cancel();
    _collector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: const Color(0xFFF1F5F9),
    body: SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(28),
        child: Column(children: [
          _buildTopBar(context),
          const SizedBox(height: 18),
          _buildHeadline(),
          const Spacer(),
          _buildStage(),
          const Spacer(),
          _buildFooter(),
        ]),
      ),
    ),
  );

  Widget _buildTopBar(BuildContext context) => Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
    IconButton(onPressed: () => Navigator.of(context).pop(), icon: const Icon(Icons.arrow_back, color: Color(0xFF0F172A))),
    _buildStatusPill(),
  ]);

  Widget _buildStatusPill() {
    final text = _phase == CollectionPhase.recording ? 'Recording' : _phase == CollectionPhase.countdown ? 'Get Ready' : _phase == CollectionPhase.complete ? 'Complete' : 'Idle';
    final color = _phase == CollectionPhase.recording ? Colors.green : const Color(0xFF6366F1);
    return Container(padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6), decoration: BoxDecoration(color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(20)), child: Text(text, style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: color)));
  }

  Widget _buildHeadline() => Column(children: [
    Text('Labeled Collection', style: GoogleFonts.outfit(fontSize: 26, fontWeight: FontWeight.w900, color: const Color(0xFF0F172A))),
    const SizedBox(height: 6),
    Text('2s windows • 50Hz • 5 minutes', style: GoogleFonts.outfit(fontSize: 13, color: const Color(0xFF64748B))),
  ]);

  Widget _buildStage() => AnimatedSwitcher(
    duration: const Duration(milliseconds: 250),
    child: switch (_phase) {
      CollectionPhase.idle => _buildStartCard(),
      CollectionPhase.chooseLabel => _buildLabelPicker(),
      CollectionPhase.countdown => _buildCountdown(),
      CollectionPhase.recording => _buildRecording(),
      CollectionPhase.complete => _buildComplete(),
    },
  );

  Widget _buildStartCard() => _buildPrimaryCard(
    title: 'Start data collection',
    subtitle: 'Label a 5-minute session',
    buttonText: 'Start data collection',
    onTap: () => setState(() => _phase = CollectionPhase.chooseLabel),
  );

  Widget _buildLabelPicker() => Column(children: [
    Text('Select activity', style: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.w700, color: const Color(0xFF0F172A))),
    const SizedBox(height: 14),
    Wrap(spacing: 12, runSpacing: 12, children: [
      _buildLabelButton('Walking'),
      _buildLabelButton('Jogging'),
      _buildLabelButton('Standing'),
      _buildLabelButton('Sitting'),
    ]),
  ]);

  Widget _buildLabelButton(String label) => _buildGradientButton('Start $label', onTap: () => _startCountdown(label));

  Widget _buildCountdown() => Column(children: [
    Text('Get ready', style: GoogleFonts.outfit(fontSize: 16, fontWeight: FontWeight.w600, color: const Color(0xFF64748B))),
    const SizedBox(height: 10),
    Text('$_countdown', style: GoogleFonts.outfit(fontSize: 64, fontWeight: FontWeight.w900, color: const Color(0xFF0F172A))),
    const SizedBox(height: 8),
    Text('Place the phone in your pocket', style: GoogleFonts.outfit(fontSize: 13, color: const Color(0xFF64748B))),
  ]);

  Widget _buildRecording() {
    final progress = (_elapsedSeconds / _recordDuration.inSeconds).clamp(0.0, 1.0);
    return Column(children: [
      Text(_label.toUpperCase(), style: GoogleFonts.outfit(fontSize: 30, fontWeight: FontWeight.w900, color: const Color(0xFF0F172A))),
      const SizedBox(height: 6),
      Text('${_formatTime(_elapsedSeconds)} / ${_formatTime(_recordDuration.inSeconds)}', style: GoogleFonts.outfit(fontSize: 14, fontWeight: FontWeight.w600, color: const Color(0xFF6366F1))),
      const SizedBox(height: 14),
      ClipRRect(borderRadius: BorderRadius.circular(999), child: LinearProgressIndicator(value: progress, minHeight: 8, backgroundColor: Colors.black12, color: const Color(0xFF6366F1))),
      const SizedBox(height: 12),
      Text('Recording… keep steady movement', style: GoogleFonts.outfit(fontSize: 12, color: const Color(0xFF64748B))),
    ]);
  }

  Widget _buildComplete() => Column(children: [
    Text('Data Collected for $_label', style: GoogleFonts.outfit(fontSize: 20, fontWeight: FontWeight.w800, color: const Color(0xFF0F172A))),
    const SizedBox(height: 6),
    Text('Windows: ${_result?.windows.length ?? 0}', style: GoogleFonts.outfit(fontSize: 12, color: const Color(0xFF64748B))),
    const SizedBox(height: 18),
    Row(children: [
      Expanded(child: _buildGradientButton('Save', onTap: _busy ? null : () => _save(popAfter: true))),
      const SizedBox(width: 12),
      Expanded(child: _buildGhostButton('Discard', onTap: _busy ? null : _discard)),
    ]),
    const SizedBox(height: 12),
    _buildGhostButton('Export / Share', onTap: _busy ? null : _exportShare),
  ]);

  Widget _buildFooter() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(18)),
    child: Row(children: [
      const Icon(Icons.shield_moon, size: 16, color: Color(0xFF64748B)),
      const SizedBox(width: 8),
      Expanded(child: Text('Orientation hint: pocket_portrait', style: GoogleFonts.outfit(fontSize: 12, color: const Color(0xFF64748B)))),
    ]),
  );

  Widget _buildPrimaryCard({required String title, required String subtitle, required String buttonText, required VoidCallback onTap}) => Container(
    padding: const EdgeInsets.all(20),
    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(22)),
    child: Column(children: [
      Text(title, style: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.w800, color: const Color(0xFF0F172A))),
      const SizedBox(height: 6),
      Text(subtitle, style: GoogleFonts.outfit(fontSize: 13, color: const Color(0xFF64748B))),
      const SizedBox(height: 16),
      _buildGradientButton(buttonText, onTap: onTap),
    ]),
  );

  Widget _buildGradientButton(String text, {VoidCallback? onTap}) => GestureDetector(
    onTap: onTap,
    child: Opacity(
      opacity: onTap == null ? 0.5 : 1,
      child: Container(
        height: 54,
        alignment: Alignment.center,
        decoration: BoxDecoration(borderRadius: BorderRadius.circular(18), gradient: const LinearGradient(colors: [Color(0xFF6366F1), Color(0xFF4F46E5)]), boxShadow: [BoxShadow(color: const Color(0xFF6366F1).withOpacity(0.3), blurRadius: 18, offset: const Offset(0, 10))]),
        child: Text(text, style: GoogleFonts.outfit(fontSize: 16, fontWeight: FontWeight.w800, color: Colors.white)),
      ),
    ),
  );

  Widget _buildGhostButton(String text, {VoidCallback? onTap}) => GestureDetector(
    onTap: onTap,
    child: Opacity(
      opacity: onTap == null ? 0.5 : 1,
      child: Container(
        height: 54,
        alignment: Alignment.center,
        decoration: BoxDecoration(borderRadius: BorderRadius.circular(18), color: Colors.white, border: Border.all(color: Colors.black12)),
        child: Text(text, style: GoogleFonts.outfit(fontSize: 15, fontWeight: FontWeight.w700, color: const Color(0xFF0F172A))),
      ),
    ),
  );

  void _startCountdown(String label) {
    _label = label;
    _countdown = _countdownSeconds;
    _phase = CollectionPhase.countdown;
    _countdownTimer?.cancel();
    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted) return;
      setState(() => _countdown--);
      if (_countdown <= 0) { t.cancel(); _beginRecording(); }
    });
    setState(() {});
  }

  void _beginRecording() {
    _elapsedSeconds = 0;
    _phase = CollectionPhase.recording;
    _collector.start(_label);
    _elapsedTimer?.cancel();
    _elapsedTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted) return;
      setState(() => _elapsedSeconds++);
    });
    _autoStopTimer?.cancel();
    _autoStopTimer = Timer(_recordDuration, _stopRecording);
    setState(() {});
  }

  void _stopRecording() {
    if (_phase != CollectionPhase.recording) return;
    _elapsedTimer?.cancel();
    _autoStopTimer?.cancel();
    _result = _collector.stop(duration: _recordDuration, orientationHint: 'pocket_portrait');
    setState(() => _phase = CollectionPhase.complete);
  }

  Future<void> _save({required bool popAfter}) async {
    if (_result == null || _busy) return;
    setState(() => _busy = true);
    final saved = await _exporter.saveResult(_result!);
    if (!mounted) return;
    setState(() { _saved = saved; _busy = false; });
    _showSnack('Saved to ${saved.csvFile.path}');
    if (popAfter) Navigator.of(context).pop();
  }

  Future<void> _exportShare() async {
    if (_result == null || _busy) return;
    setState(() => _busy = true);
    final saved = _saved ?? await _exporter.saveResult(_result!);
    if (!mounted) return;
    _saved = saved;
    setState(() => _busy = false);
    if (!mounted) return;
    await showModalBottomSheet(context: context, shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(24))), builder: (_) => _buildExportSheet(saved));
  }

  Widget _buildExportSheet(CollectionSaveResult saved) => Padding(
    padding: const EdgeInsets.all(20),
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      Text('Export Dataset', style: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.w800)),
      const SizedBox(height: 12),
      _buildGradientButton('Share Zip', onTap: () => _shareZip(saved)),
      const SizedBox(height: 10),
      _buildGhostButton('Upload to Server', onTap: () => _upload(saved)),
      const SizedBox(height: 8),
    ]),
  );

  Future<void> _shareZip(CollectionSaveResult saved) async {
    Navigator.of(context).pop();
    setState(() => _busy = true);
    final zip = await _exporter.createZip(saved);
    if (!mounted) return;
    setState(() => _busy = false);
    await Share.shareXFiles([XFile(zip.path)], text: 'HAR dataset');
  }

  Future<void> _upload(CollectionSaveResult saved) async {
    Navigator.of(context).pop();
    setState(() => _busy = true);
    final ok = await _api.uploadDataset(csvFile: saved.csvFile, summaryFile: saved.summaryFile);
    if (!mounted) return;
    setState(() => _busy = false);
    _showSnack(ok ? 'Upload complete' : 'Upload failed');
  }

  void _discard() => Navigator.of(context).pop();

  void _showSnack(String text) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));

  String _formatTime(int seconds) => '${(seconds ~/ 60).toString().padLeft(2, '0')}:${(seconds % 60).toString().padLeft(2, '0')}';
}
