"""Trusted sandbox policy validation tests."""

import unittest

from pydantic import ValidationError

from sandbox.policy import SandboxPolicy


class SandboxPolicyTests(unittest.TestCase):
    def test_secure_docker_defaults_are_not_disableable(self) -> None:
        policy = SandboxPolicy(backend="docker")
        self.assertFalse(policy.network_enabled)
        self.assertTrue(policy.read_only_root)
        self.assertTrue(policy.cap_drop_all)
        self.assertTrue(policy.no_new_privileges)
        for update in (
            {"network_enabled": True}, {"read_only_root": False},
            {"cap_drop_all": False}, {"no_new_privileges": False},
        ):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                SandboxPolicy(backend="docker", **update)

    def test_resource_limits_reject_zero_and_hard_maximum(self) -> None:
        for update in (
            {"memory_limit_mb": 0}, {"memory_limit_mb": 4097},
            {"cpu_limit": 0}, {"cpu_limit": 5},
            {"pids_limit": 0}, {"pids_limit": 257},
            {"max_timeout_seconds": 0}, {"max_timeout_seconds": 3601},
        ):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                SandboxPolicy(**update)

    def test_image_is_one_reference_and_environment_is_fixed(self) -> None:
        with self.assertRaises(ValidationError):
            SandboxPolicy(image="image --privileged")
        with self.assertRaises(ValidationError):
            SandboxPolicy(allowed_environment={"OPENAI_API_KEY"})


if __name__ == "__main__":
    unittest.main()
