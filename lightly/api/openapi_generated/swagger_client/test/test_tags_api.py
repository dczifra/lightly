"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://openapi-generator.tech
"""


import unittest

import lightly.api.openapi_generated.swagger_client
from lightly.api.openapi_generated.swagger_client.api.tags_api import TagsApi  # noqa: E501


class TestTagsApi(unittest.TestCase):
    """TagsApi unit test stubs"""

    def setUp(self):
        self.api = TagsApi()  # noqa: E501

    def tearDown(self):
        pass

    def test_create_initial_tag_by_dataset_id(self):
        """Test case for create_initial_tag_by_dataset_id

        """
        pass

    def test_create_tag_by_dataset_id(self):
        """Test case for create_tag_by_dataset_id

        """
        pass

    def test_delete_tag_by_tag_id(self):
        """Test case for delete_tag_by_tag_id

        """
        pass

    def test_export_tag_to_label_box_data_rows(self):
        """Test case for export_tag_to_label_box_data_rows

        """
        pass

    def test_export_tag_to_label_studio_tasks(self):
        """Test case for export_tag_to_label_studio_tasks

        """
        pass

    def test_get_filenames_by_tag_id(self):
        """Test case for get_filenames_by_tag_id

        """
        pass

    def test_get_tag_by_tag_id(self):
        """Test case for get_tag_by_tag_id

        """
        pass

    def test_get_tags_by_dataset_id(self):
        """Test case for get_tags_by_dataset_id

        """
        pass

    def test_perform_tag_arithmetics(self):
        """Test case for perform_tag_arithmetics

        """
        pass

    def test_perform_tag_arithmetics_bitmask(self):
        """Test case for perform_tag_arithmetics_bitmask

        """
        pass

    def test_update_tag_by_tag_id(self):
        """Test case for update_tag_by_tag_id

        """
        pass

    def test_upsize_tags_by_dataset_id(self):
        """Test case for upsize_tags_by_dataset_id

        """
        pass


if __name__ == '__main__':
    unittest.main()