import unittest

from transclip.mode_routing import route_voice_mode


class ModeRoutingTests(unittest.TestCase):
    def test_case_insensitive_cleanup_and_shell_prefixes(self):
        cleanup = route_voice_mode("Clean up hello world")
        shell = route_voice_mode("Shell command, list files")

        self.assertEqual(cleanup.mode, "cleanup")
        self.assertEqual(cleanup.trigger, "clean up")
        self.assertEqual(cleanup.payload, "hello world")
        self.assertEqual(shell.mode, "shell")
        self.assertEqual(shell.trigger, "shell command")
        self.assertEqual(shell.payload, "list files")

    def test_all_configured_triggers_match_at_prefix(self):
        self.assertEqual(route_voice_mode("trans cleanup fix spacing").mode, "cleanup")
        self.assertEqual(route_voice_mode("bash command show disk usage").mode, "shell")
        self.assertEqual(route_voice_mode("terminal command list ports").mode, "shell")

    def test_prefix_only_behavior(self):
        route = route_voice_mode("please run a shell command list files")

        self.assertEqual(route.mode, "dictation")
        self.assertEqual(route.payload, "please run a shell command list files")

    def test_literal_escape_preserves_configured_trigger_text(self):
        shell = route_voice_mode("literal shell command, list files")
        cleanup = route_voice_mode("literal clean up this sentence")
        trans_cleanup = route_voice_mode("literal trans cleanup this sentence")
        bash = route_voice_mode("literal bash command show branch")
        terminal = route_voice_mode("literal terminal command list ports")

        self.assertEqual(shell.mode, "dictation")
        self.assertTrue(shell.literal)
        self.assertEqual(shell.payload, "shell command, list files")
        self.assertEqual(cleanup.payload, "clean up this sentence")
        self.assertEqual(trans_cleanup.payload, "trans cleanup this sentence")
        self.assertEqual(bash.payload, "bash command show branch")
        self.assertEqual(terminal.payload, "terminal command list ports")

    def test_empty_trigger_payload_falls_back_to_dictation(self):
        cleanup = route_voice_mode("clean up")
        shell = route_voice_mode("bash command,")

        self.assertEqual(cleanup.mode, "dictation")
        self.assertEqual(cleanup.payload, "clean up")
        self.assertEqual(shell.mode, "dictation")
        self.assertEqual(shell.payload, "bash command,")

    def test_original_payload_preservation(self):
        route = route_voice_mode("Clean up   keep /tmp/MyFile --flag")

        self.assertEqual(route.payload, "keep /tmp/MyFile --flag")

    def test_disabled_shell_trigger_falls_back_to_dictation(self):
        route = route_voice_mode("shell command list files", shell_enabled=False)

        self.assertEqual(route.mode, "dictation")
        self.assertEqual(route.payload, "shell command list files")


if __name__ == "__main__":
    unittest.main()
