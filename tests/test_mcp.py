import unittest
from unittest.mock import patch

from knowledge_base import mcp_server


class McpTests(unittest.TestCase):
    def test_tools_expose_only_safe_read_operations(self):
        tools = {item["name"]: item for item in mcp_server._tools()}

        self.assertEqual({"search_knowledge", "list_knowledge", "list_knowledge_files"}, set(tools))
        self.assertIn("query", tools["search_knowledge"]["inputSchema"]["required"])
        self.assertNotIn("stored_path", str(tools))

    def test_search_tool_validates_arguments(self):
        with self.assertRaises(ValueError):
            mcp_server._call_tool("search_knowledge", {"query": ""})
        with self.assertRaises(ValueError):
            mcp_server._call_tool("search_knowledge", {"query": "hello", "n_results": 0})
        with self.assertRaises(ValueError):
            mcp_server._call_tool("search_knowledge", {"query": "hello", "n_results": True})

    def test_search_tool_delegates_to_safe_store_api(self):
        expected = [{"content": "safe", "metadata": {"source_id": "doc-1"}}]
        with patch.object(mcp_server, "search_documents", return_value=expected) as search:
            result = mcp_server._call_tool("search_knowledge", {"query": "hello", "n_results": 2})

        self.assertEqual(expected, result)
        search.assert_called_once_with("hello", 2)

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            mcp_server._call_tool("delete_everything", {})


if __name__ == "__main__":
    unittest.main()
