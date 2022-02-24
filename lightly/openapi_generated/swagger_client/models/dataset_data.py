# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from lightly.openapi_generated.swagger_client.configuration import Configuration


class DatasetData(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'id': 'MongoObjectID',
        'name': 'DatasetName',
        'user_id': 'str',
        'access_type': 'SharedAccessType',
        'type': 'DatasetType',
        'img_type': 'ImageType',
        'n_samples': 'int',
        'size_in_bytes': 'int',
        'meta_data_configuration_id': 'MongoObjectID',
        'created_at': 'Timestamp',
        'last_modified_at': 'Timestamp'
    }

    attribute_map = {
        'id': 'id',
        'name': 'name',
        'user_id': 'userId',
        'access_type': 'accessType',
        'type': 'type',
        'img_type': 'imgType',
        'n_samples': 'nSamples',
        'size_in_bytes': 'sizeInBytes',
        'meta_data_configuration_id': 'metaDataConfigurationId',
        'created_at': 'createdAt',
        'last_modified_at': 'lastModifiedAt'
    }

    def __init__(self, id=None, name=None, user_id=None, access_type=None, type=None, img_type=None, n_samples=None, size_in_bytes=None, meta_data_configuration_id=None, created_at=None, last_modified_at=None, _configuration=None):  # noqa: E501
        """DatasetData - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._id = None
        self._name = None
        self._user_id = None
        self._access_type = None
        self._type = None
        self._img_type = None
        self._n_samples = None
        self._size_in_bytes = None
        self._meta_data_configuration_id = None
        self._created_at = None
        self._last_modified_at = None
        self.discriminator = None

        self.id = id
        self.name = name
        self.user_id = user_id
        if access_type is not None:
            self.access_type = access_type
        self.type = type
        if img_type is not None:
            self.img_type = img_type
        self.n_samples = n_samples
        self.size_in_bytes = size_in_bytes
        if meta_data_configuration_id is not None:
            self.meta_data_configuration_id = meta_data_configuration_id
        self.created_at = created_at
        self.last_modified_at = last_modified_at

    @property
    def id(self):
        """Gets the id of this DatasetData.  # noqa: E501


        :return: The id of this DatasetData.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this DatasetData.


        :param id: The id of this DatasetData.  # noqa: E501
        :type: MongoObjectID
        """
        if self._configuration.client_side_validation and id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def name(self):
        """Gets the name of this DatasetData.  # noqa: E501


        :return: The name of this DatasetData.  # noqa: E501
        :rtype: DatasetName
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this DatasetData.


        :param name: The name of this DatasetData.  # noqa: E501
        :type: DatasetName
        """
        if self._configuration.client_side_validation and name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def user_id(self):
        """Gets the user_id of this DatasetData.  # noqa: E501

        The owner of the dataset  # noqa: E501

        :return: The user_id of this DatasetData.  # noqa: E501
        :rtype: str
        """
        return self._user_id

    @user_id.setter
    def user_id(self, user_id):
        """Sets the user_id of this DatasetData.

        The owner of the dataset  # noqa: E501

        :param user_id: The user_id of this DatasetData.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and user_id is None:
            raise ValueError("Invalid value for `user_id`, must not be `None`")  # noqa: E501

        self._user_id = user_id

    @property
    def access_type(self):
        """Gets the access_type of this DatasetData.  # noqa: E501


        :return: The access_type of this DatasetData.  # noqa: E501
        :rtype: SharedAccessType
        """
        return self._access_type

    @access_type.setter
    def access_type(self, access_type):
        """Sets the access_type of this DatasetData.


        :param access_type: The access_type of this DatasetData.  # noqa: E501
        :type: SharedAccessType
        """

        self._access_type = access_type

    @property
    def type(self):
        """Gets the type of this DatasetData.  # noqa: E501


        :return: The type of this DatasetData.  # noqa: E501
        :rtype: DatasetType
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this DatasetData.


        :param type: The type of this DatasetData.  # noqa: E501
        :type: DatasetType
        """
        if self._configuration.client_side_validation and type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")  # noqa: E501

        self._type = type

    @property
    def img_type(self):
        """Gets the img_type of this DatasetData.  # noqa: E501


        :return: The img_type of this DatasetData.  # noqa: E501
        :rtype: ImageType
        """
        return self._img_type

    @img_type.setter
    def img_type(self, img_type):
        """Sets the img_type of this DatasetData.


        :param img_type: The img_type of this DatasetData.  # noqa: E501
        :type: ImageType
        """

        self._img_type = img_type

    @property
    def n_samples(self):
        """Gets the n_samples of this DatasetData.  # noqa: E501


        :return: The n_samples of this DatasetData.  # noqa: E501
        :rtype: int
        """
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n_samples):
        """Sets the n_samples of this DatasetData.


        :param n_samples: The n_samples of this DatasetData.  # noqa: E501
        :type: int
        """
        if self._configuration.client_side_validation and n_samples is None:
            raise ValueError("Invalid value for `n_samples`, must not be `None`")  # noqa: E501

        self._n_samples = n_samples

    @property
    def size_in_bytes(self):
        """Gets the size_in_bytes of this DatasetData.  # noqa: E501


        :return: The size_in_bytes of this DatasetData.  # noqa: E501
        :rtype: int
        """
        return self._size_in_bytes

    @size_in_bytes.setter
    def size_in_bytes(self, size_in_bytes):
        """Sets the size_in_bytes of this DatasetData.


        :param size_in_bytes: The size_in_bytes of this DatasetData.  # noqa: E501
        :type: int
        """
        if self._configuration.client_side_validation and size_in_bytes is None:
            raise ValueError("Invalid value for `size_in_bytes`, must not be `None`")  # noqa: E501

        self._size_in_bytes = size_in_bytes

    @property
    def meta_data_configuration_id(self):
        """Gets the meta_data_configuration_id of this DatasetData.  # noqa: E501


        :return: The meta_data_configuration_id of this DatasetData.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._meta_data_configuration_id

    @meta_data_configuration_id.setter
    def meta_data_configuration_id(self, meta_data_configuration_id):
        """Sets the meta_data_configuration_id of this DatasetData.


        :param meta_data_configuration_id: The meta_data_configuration_id of this DatasetData.  # noqa: E501
        :type: MongoObjectID
        """

        self._meta_data_configuration_id = meta_data_configuration_id

    @property
    def created_at(self):
        """Gets the created_at of this DatasetData.  # noqa: E501


        :return: The created_at of this DatasetData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at):
        """Sets the created_at of this DatasetData.


        :param created_at: The created_at of this DatasetData.  # noqa: E501
        :type: Timestamp
        """
        if self._configuration.client_side_validation and created_at is None:
            raise ValueError("Invalid value for `created_at`, must not be `None`")  # noqa: E501

        self._created_at = created_at

    @property
    def last_modified_at(self):
        """Gets the last_modified_at of this DatasetData.  # noqa: E501


        :return: The last_modified_at of this DatasetData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._last_modified_at

    @last_modified_at.setter
    def last_modified_at(self, last_modified_at):
        """Sets the last_modified_at of this DatasetData.


        :param last_modified_at: The last_modified_at of this DatasetData.  # noqa: E501
        :type: Timestamp
        """
        if self._configuration.client_side_validation and last_modified_at is None:
            raise ValueError("Invalid value for `last_modified_at`, must not be `None`")  # noqa: E501

        self._last_modified_at = last_modified_at

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(DatasetData, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, DatasetData):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, DatasetData):
            return True

        return self.to_dict() != other.to_dict()
