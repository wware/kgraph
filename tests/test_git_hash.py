"""Tests for git hash utility in export module.

Tests the get_git_hash() function for version tracking in bundles.
"""

import subprocess
from unittest.mock import patch, MagicMock


from kgraph.export import get_git_hash


class TestGetGitHash:
    """Test the get_git_hash() function."""

    def test_returns_string_in_git_repo(self):
        """Should return a string when in a git repository."""
        # This test runs in the actual kgraph repo, so it should work
        result = get_git_hash()
        # We're in a git repo, so this should return a hash
        assert result is not None
        assert isinstance(result, str)

    def test_returns_short_hash_format(self):
        """Hash should be short format (7+ characters, alphanumeric)."""
        result = get_git_hash()
        if result is not None:
            # Short hash is typically 7-10 characters
            assert len(result) >= 7
            assert len(result) <= 12
            # Should be alphanumeric (hex)
            assert all(c in "0123456789abcdef" for c in result.lower())

    def test_returns_none_when_git_unavailable(self):
        """Should return None when git command fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            result = get_git_hash()
        assert result is None

    def test_returns_none_when_not_in_repo(self):
        """Should return None when not in a git repository."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=128,
                cmd=["git", "rev-parse", "--short", "HEAD"],
            )
            result = get_git_hash()
        assert result is None

    def test_returns_none_on_timeout(self):
        """Should return None when git command times out."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["git", "rev-parse", "--short", "HEAD"],
                timeout=5.0,
            )
            result = get_git_hash()
        assert result is None

    def test_strips_whitespace_from_output(self):
        """Should strip whitespace from git output."""
        mock_result = MagicMock()
        mock_result.stdout = "  abc1234\n"

        with patch("subprocess.run", return_value=mock_result):
            result = get_git_hash()

        assert result == "abc1234"

    def test_uses_correct_git_command(self):
        """Should call git rev-parse --short HEAD."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "abc1234"
            mock_run.return_value = mock_result

            get_git_hash()

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == ["git", "rev-parse", "--short", "HEAD"]
            assert call_args[1]["capture_output"] is True
            assert call_args[1]["text"] is True
            assert call_args[1]["check"] is True
            assert call_args[1]["timeout"] == 5.0
