import unittest

from transclip import product


class ProductTests(unittest.TestCase):
    def test_product_identity_constants_are_transclip(self):
        self.assertEqual(product.DISPLAY_NAME, "TransClip")
        self.assertEqual(product.APP_ID, "transclip")
        self.assertEqual(product.CLI_COMMAND, "transclip")
        self.assertEqual(product.SERVICE_NAME, "transclip.service")
        self.assertEqual(product.LAUNCHD_LABEL, "com.paulbrav.transclip")
        self.assertEqual(product.CONFIG_DIR_NAME, "transclip")
        self.assertEqual(product.CACHE_DIR_NAME, "transclip")
        self.assertEqual(product.LOG_DIR_NAME, "transclip")
        self.assertEqual(product.SHORTCUT_NAME, "TransClip Toggle")
        self.assertEqual(product.IMPORT_PACKAGE, "transclip")


if __name__ == "__main__":
    unittest.main()
