import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "https://dev.turaapp.com";

  Future<Map<String, dynamic>?> predictActivity(List<List<double>> samples) async {
    final payload = {
      "sampling_rate": 50,
      "timestamp": DateTime.now().millisecondsSinceEpoch ~/ 1000, // Unix timestamp in seconds
      "samples": samples,
    };

    for (int i = 0; i < 3; i++) {
      try {
        final res = await http.post(
          Uri.parse('$baseUrl/har/'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload),
        ).timeout(const Duration(seconds: 4));
        if (res.statusCode == 200) return jsonDecode(res.body);
      } catch (e) { print('Retry ${i+1} failed: $e'); }
    }
    return null;
  }
}
