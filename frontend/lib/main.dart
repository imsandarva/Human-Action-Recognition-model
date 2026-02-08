import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:har_app/screens/home_screen.dart';

void main() => runApp(const HarApp());

class HarApp extends StatelessWidget {
  const HarApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6366F1)),
      textTheme: GoogleFonts.outfitTextTheme(),
    ),
    home: const HomeScreen(),
  );
}
