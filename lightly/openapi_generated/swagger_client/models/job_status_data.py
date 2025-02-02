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


class JobStatusData(object):
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
        'status': 'JobState',
        'meta': 'JobStatusMeta',
        'wait_time_till_next_poll': 'int',
        'created_at': 'Timestamp',
        'last_modified_at': 'Timestamp',
        'finished_at': 'Timestamp',
        'error': 'str',
        'result': 'JobStatusDataResult'
    }

    attribute_map = {
        'id': 'id',
        'status': 'status',
        'meta': 'meta',
        'wait_time_till_next_poll': 'waitTimeTillNextPoll',
        'created_at': 'createdAt',
        'last_modified_at': 'lastModifiedAt',
        'finished_at': 'finishedAt',
        'error': 'error',
        'result': 'result'
    }

    def __init__(self, id=None, status=None, meta=None, wait_time_till_next_poll=None, created_at=None, last_modified_at=None, finished_at=None, error=None, result=None, _configuration=None):  # noqa: E501
        """JobStatusData - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._id = None
        self._status = None
        self._meta = None
        self._wait_time_till_next_poll = None
        self._created_at = None
        self._last_modified_at = None
        self._finished_at = None
        self._error = None
        self._result = None
        self.discriminator = None

        self.id = id
        self.status = status
        if meta is not None:
            self.meta = meta
        self.wait_time_till_next_poll = wait_time_till_next_poll
        self.created_at = created_at
        if last_modified_at is not None:
            self.last_modified_at = last_modified_at
        if finished_at is not None:
            self.finished_at = finished_at
        if error is not None:
            self.error = error
        if result is not None:
            self.result = result

    @property
    def id(self):
        """Gets the id of this JobStatusData.  # noqa: E501


        :return: The id of this JobStatusData.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this JobStatusData.


        :param id: The id of this JobStatusData.  # noqa: E501
        :type: MongoObjectID
        """
        if self._configuration.client_side_validation and id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def status(self):
        """Gets the status of this JobStatusData.  # noqa: E501


        :return: The status of this JobStatusData.  # noqa: E501
        :rtype: JobState
        """
        return self._status

    @status.setter
    def status(self, status):
        """Sets the status of this JobStatusData.


        :param status: The status of this JobStatusData.  # noqa: E501
        :type: JobState
        """
        if self._configuration.client_side_validation and status is None:
            raise ValueError("Invalid value for `status`, must not be `None`")  # noqa: E501

        self._status = status

    @property
    def meta(self):
        """Gets the meta of this JobStatusData.  # noqa: E501


        :return: The meta of this JobStatusData.  # noqa: E501
        :rtype: JobStatusMeta
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """Sets the meta of this JobStatusData.


        :param meta: The meta of this JobStatusData.  # noqa: E501
        :type: JobStatusMeta
        """

        self._meta = meta

    @property
    def wait_time_till_next_poll(self):
        """Gets the wait_time_till_next_poll of this JobStatusData.  # noqa: E501

        The time in seconds the client should wait before doing the next poll.  # noqa: E501

        :return: The wait_time_till_next_poll of this JobStatusData.  # noqa: E501
        :rtype: int
        """
        return self._wait_time_till_next_poll

    @wait_time_till_next_poll.setter
    def wait_time_till_next_poll(self, wait_time_till_next_poll):
        """Sets the wait_time_till_next_poll of this JobStatusData.

        The time in seconds the client should wait before doing the next poll.  # noqa: E501

        :param wait_time_till_next_poll: The wait_time_till_next_poll of this JobStatusData.  # noqa: E501
        :type: int
        """
        if self._configuration.client_side_validation and wait_time_till_next_poll is None:
            raise ValueError("Invalid value for `wait_time_till_next_poll`, must not be `None`")  # noqa: E501

        self._wait_time_till_next_poll = wait_time_till_next_poll

    @property
    def created_at(self):
        """Gets the created_at of this JobStatusData.  # noqa: E501


        :return: The created_at of this JobStatusData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at):
        """Sets the created_at of this JobStatusData.


        :param created_at: The created_at of this JobStatusData.  # noqa: E501
        :type: Timestamp
        """
        if self._configuration.client_side_validation and created_at is None:
            raise ValueError("Invalid value for `created_at`, must not be `None`")  # noqa: E501

        self._created_at = created_at

    @property
    def last_modified_at(self):
        """Gets the last_modified_at of this JobStatusData.  # noqa: E501


        :return: The last_modified_at of this JobStatusData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._last_modified_at

    @last_modified_at.setter
    def last_modified_at(self, last_modified_at):
        """Sets the last_modified_at of this JobStatusData.


        :param last_modified_at: The last_modified_at of this JobStatusData.  # noqa: E501
        :type: Timestamp
        """

        self._last_modified_at = last_modified_at

    @property
    def finished_at(self):
        """Gets the finished_at of this JobStatusData.  # noqa: E501


        :return: The finished_at of this JobStatusData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._finished_at

    @finished_at.setter
    def finished_at(self, finished_at):
        """Sets the finished_at of this JobStatusData.


        :param finished_at: The finished_at of this JobStatusData.  # noqa: E501
        :type: Timestamp
        """

        self._finished_at = finished_at

    @property
    def error(self):
        """Gets the error of this JobStatusData.  # noqa: E501


        :return: The error of this JobStatusData.  # noqa: E501
        :rtype: str
        """
        return self._error

    @error.setter
    def error(self, error):
        """Sets the error of this JobStatusData.


        :param error: The error of this JobStatusData.  # noqa: E501
        :type: str
        """

        self._error = error

    @property
    def result(self):
        """Gets the result of this JobStatusData.  # noqa: E501


        :return: The result of this JobStatusData.  # noqa: E501
        :rtype: JobStatusDataResult
        """
        return self._result

    @result.setter
    def result(self, result):
        """Sets the result of this JobStatusData.


        :param result: The result of this JobStatusData.  # noqa: E501
        :type: JobStatusDataResult
        """

        self._result = result

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
        if issubclass(JobStatusData, dict):
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
        if not isinstance(other, JobStatusData):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, JobStatusData):
            return True

        return self.to_dict() != other.to_dict()
