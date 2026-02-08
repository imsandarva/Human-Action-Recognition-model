import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:har_app/screens/data_collection_screen.dart';
import 'package:har_app/screens/test_data_screen.dart';
import 'package:har_app/services/dataset_export_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final DatasetExportService _exporter = DatasetExportService();
  bool _exporting = false;

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: const Color(0xFFF1F5F9),
    body: SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(28),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildTitle(),
            const SizedBox(height: 12),
            Text('Pick a mode to start.', style: GoogleFonts.outfit(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.black54)),
            const Spacer(),
            _buildEntryCard(context, title: 'Collect Data', subtitle: 'Labeled 5-min sessions', icon: Icons.track_changes, colors: const [Color(0xFF6366F1), Color(0xFF4F46E5)], onTap: () => _open(context, const DataCollectionScreen())),
            const SizedBox(height: 16),
            _buildEntryCard(context, title: 'Test Data', subtitle: 'Live inference to server', icon: Icons.bolt, colors: const [Color(0xFF0EA5E9), Color(0xFF0284C7)], onTap: () => _open(context, const TestDataScreen())),
            const SizedBox(height: 16),
            _buildExportButton(),
            const Spacer(),
            _buildFooter(),
          ],
        ),
      ),
    ),
  );

  Widget _buildTitle() => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
    Text('HAR Studio', style: GoogleFonts.outfit(fontSize: 34, fontWeight: FontWeight.w900, color: const Color(0xFF0F172A), letterSpacing: -0.6)),
    Text('Pocket Portrait', style: GoogleFonts.outfit(fontSize: 14, fontWeight: FontWeight.w600, color: const Color(0xFF6366F1))),
  ]);

  Widget _buildEntryCard(BuildContext context, {required String title, required String subtitle, required IconData icon, required List<Color> colors, required VoidCallback onTap}) => GestureDetector(
    onTap: onTap,
    child: Container(
      height: 120,
      width: double.infinity,
      decoration: BoxDecoration(borderRadius: BorderRadius.circular(28), gradient: LinearGradient(colors: colors), boxShadow: [BoxShadow(color: colors.first.withOpacity(0.35), blurRadius: 24, offset: const Offset(0, 12))]),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
        child: Row(
          children: [
            Container(width: 56, height: 56, decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(18)), child: Icon(icon, color: Colors.white, size: 28)),
            const SizedBox(width: 16),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisAlignment: MainAxisAlignment.center, children: [
              Text(title, style: GoogleFonts.outfit(fontSize: 20, fontWeight: FontWeight.w800, color: Colors.white)),
              const SizedBox(height: 6),
              Text(subtitle, style: GoogleFonts.outfit(fontSize: 13, fontWeight: FontWeight.w500, color: Colors.white70)),
            ])),
            const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white70),
          ],
        ),
      ),
    ),
  );

  Widget _buildExportButton() => GestureDetector(
    onTap: _exporting ? null : _exportAllData,
    child: Container(
      height: 64,
      width: double.infinity,
      decoration: BoxDecoration(borderRadius: BorderRadius.circular(20), color: Colors.white, border: Border.all(color: const Color(0xFFE2E8F0), width: 1.5), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: Row(children: [
          Icon(_exporting ? Icons.hourglass_empty : Icons.download_rounded, size: 22, color: _exporting ? const Color(0xFF94A3B8) : const Color(0xFF6366F1)),
          const SizedBox(width: 12),
          Expanded(child: Text(_exporting ? 'Exporting...' : 'Export All Data', style: GoogleFonts.outfit(fontSize: 15, fontWeight: FontWeight.w700, color: _exporting ? const Color(0xFF94A3B8) : const Color(0xFF0F172A)))),
          if (!_exporting) const Icon(Icons.arrow_forward_ios, size: 14, color: Color(0xFF94A3B8)),
        ]),
      ),
    ),
  );

  Widget _buildFooter() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(18)),
    child: Row(children: [
      const Icon(Icons.phone_iphone, size: 16, color: Color(0xFF64748B)),
      const SizedBox(width: 8),
      Expanded(child: Text('Keep the phone in pocket, screen facing body.', style: GoogleFonts.outfit(fontSize: 12, color: const Color(0xFF64748B)))),
    ]),
);

  Future<void> _exportAllData() async {
    setState(() => _exporting = true);
    final path = await _exporter.exportAllAppData();
    if (!mounted) return;
    setState(() => _exporting = false);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(path != null ? 'Exported to: $path' : 'Export failed or no data found', style: GoogleFonts.outfit())));
  }

  void _open(BuildContext context, Widget screen) => Navigator.of(context).push(MaterialPageRoute(builder: (_) => screen));
}
