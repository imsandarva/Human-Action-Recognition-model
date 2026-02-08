import 'dart:convert';
import 'dart:io';
import 'package:archive/archive.dart';
import 'package:intl/intl.dart';
import 'package:path_provider/path_provider.dart';
import 'package:har_app/services/data_collection_service.dart';

class CollectionSaveResult {
  final File csvFile;
  final File summaryFile;
  final Map<String, dynamic> summaryJson;
  final String timestamp;
  const CollectionSaveResult({required this.csvFile, required this.summaryFile, required this.summaryJson, required this.timestamp});
}

class DatasetExportService {
  Future<CollectionSaveResult> saveResult(DataCollectionResult result) async {
    final ts = DateFormat('yyyyMMdd_HHmmss').format(DateTime.now());
    final label = _sanitizeLabel(result.label);
    final csvName = '${label}_collection_$ts.csv';
    final summaryName = 'summary_$ts.json';
    final dir = await getApplicationDocumentsDirectory();
    final csvFile = File('${dir.path}/$csvName');
    final summaryFile = File('${dir.path}/$summaryName');
    await csvFile.writeAsString(_buildCsv(result), flush: true);
    final summaryJson = _buildSummary(result, csvName);
    await summaryFile.writeAsString(jsonEncode(summaryJson), flush: true);
    return CollectionSaveResult(csvFile: csvFile, summaryFile: summaryFile, summaryJson: summaryJson, timestamp: ts);
  }

  Future<File> createZip(CollectionSaveResult save) async {
    final archive = Archive()
      ..addFile(ArchiveFile(save.csvFile.uri.pathSegments.last, save.csvFile.lengthSync(), save.csvFile.readAsBytesSync()))
      ..addFile(ArchiveFile(save.summaryFile.uri.pathSegments.last, save.summaryFile.lengthSync(), save.summaryFile.readAsBytesSync()));
    final zipBytes = ZipEncoder().encode(archive);
    final dir = await getTemporaryDirectory();
    final zipFile = File('${dir.path}/dataset_${save.timestamp}.zip');
    await zipFile.writeAsBytes(zipBytes, flush: true);
    return zipFile;
  }

  String _buildCsv(DataCollectionResult result) {
    final b = StringBuffer('window_id,label,start_ts_ms,avg_rate,sample_index,sample_ts_ms,ax,ay,az\n');
    for (final w in result.windows) {
      for (var i = 0; i < w.samples.length; i++) {
        final s = w.samples[i];
        b.writeln('${w.id},${w.label},${w.startTsMs},${w.avgRate.toStringAsFixed(2)},$i,${s.tsMs},${s.ax},${s.ay},${s.az}');
      }
    }
    return b.toString();
  }

  Map<String, dynamic> _buildSummary(DataCollectionResult result, String csvName) {
    final summary = <String, dynamic>{
      'label': result.label,
      'total_windows': result.windows.length,
      'sampling_rate_requested': result.samplingRateRequested,
      'actual_avg_rate_over_all': double.parse(result.avgRateOverAll.toStringAsFixed(2)),
      'duration_s': result.duration.inSeconds,
      'filename': csvName,
      'orientation_hint': result.orientationHint,
    };
    if (result.lowRateDetected) summary['note'] = 'instantaneous_rate_below_40hz_for_over_1s';
    return summary;
  }

  String _sanitizeLabel(String label) => label.replaceAll(RegExp(r'[^a-zA-Z0-9]+'), '');

  Future<String?> exportAllAppData() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final flutterDir = Directory(appDir.path);
      if (!await flutterDir.exists()) return null;
      final files = flutterDir.listSync().whereType<File>().toList();
      if (files.isEmpty) return null;
      final archive = Archive();
      for (final file in files) {
        final name = file.uri.pathSegments.last;
        archive.addFile(ArchiveFile(name, file.lengthSync(), file.readAsBytesSync()));
      }
      final zipBytes = ZipEncoder().encode(archive);
      final downloadsPath = Platform.isAndroid ? '/storage/emulated/0/Download' : (await getExternalStorageDirectory())?.path;
      if (downloadsPath == null) return null;
      final downloadsDir = Directory(downloadsPath);
      if (!await downloadsDir.exists()) await downloadsDir.create(recursive: true);
      final zipName = 'har_app_data_${DateTime.now().millisecondsSinceEpoch}.zip';
      final zipFile = File('$downloadsPath/$zipName');
      await zipFile.writeAsBytes(zipBytes, flush: true);
      return zipFile.path;
    } catch (e) { return null; }
  }
}
