#!/usr/bin/env python3
"""
Test module for the LLM client functionality.
"""
import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock
from llm_client import answer

class TestLLMClient(unittest.TestCase):
    """Test cases for the LLM client module."""
    
    @patch('aiohttp.ClientSession.post')
    @patch('llm_client.client.calculate_cost', return_value=0.001)
    async def test_answer_with_context(self, mock_calculate_cost, mock_post):
        """Test that answer function works with provided context."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Paris is the capital of France."
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        mock_post.return_value = mock_response
        
        # Test data
        question = "What is the capital of France?"
        context = ["Paris is the capital of France.", "France is in Europe."]
        
        # Execute
        result = await answer(question, context)
        
        # Verify
        self.assertEqual(result, "Paris is the capital of France.")
        
        # Check that the API was called with the correct prompt
        mock_post.assert_called_once()
        called_args = mock_post.call_args[1]
        payload = called_args['json']
        
        # Verify that the prompt contains the context and question
        user_message = payload['messages'][1]['content']
        self.assertIn(question, user_message)
        for ctx in context:
            self.assertIn(ctx, user_message)
        
        # Verify that tools are included
        self.assertIn('tools', payload)
        self.assertEqual(payload['tools'][0]['function']['name'], 'search_web')
    
    @patch('aiohttp.ClientSession.post')
    async def test_api_error_handling(self, mock_post):
        """Test error handling when API returns non-200 status."""
        # Setup mock error response
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.__aenter__.return_value = mock_response
        mock_response.text.return_value = "Bad Request"
        
        mock_post.return_value = mock_response
        
        # Test data
        question = "What is the capital of France?"
        context = ["Paris is the capital of France."]
        
        # Execute and verify exception is raised
        with self.assertRaises(Exception) as context:
            await answer(question, context)
        
        self.assertIn("API Error: 400", str(context.exception))

def run_tests():
    """Run the test suite."""
    os.environ['OPENAI_API_KEY'] = 'fake-key-for-testing'
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLLMClient)
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    asyncio.run(unittest.main())