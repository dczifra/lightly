"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://openapi-generator.tech
"""


import re  # noqa: F401
import sys  # noqa: F401

from lightly.api.openapi_generated.swagger_client.api_client import ApiClient, Endpoint as _Endpoint
from lightly.api.openapi_generated.swagger_client.model_utils import (  # noqa: F401
    check_allowed_values,
    check_validations,
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types
)
from lightly.api.openapi_generated.swagger_client.model.api_error_response import ApiErrorResponse
from lightly.api.openapi_generated.swagger_client.model.create_entity_response import CreateEntityResponse
from lightly.api.openapi_generated.swagger_client.model.dataset_create_request import DatasetCreateRequest
from lightly.api.openapi_generated.swagger_client.model.dataset_data import DatasetData
from lightly.api.openapi_generated.swagger_client.model.dataset_data_enriched import DatasetDataEnriched
from lightly.api.openapi_generated.swagger_client.model.dataset_update_request import DatasetUpdateRequest
from lightly.api.openapi_generated.swagger_client.model.entity_body import EntityBody
from lightly.api.openapi_generated.swagger_client.model.job_status_meta import JobStatusMeta
from lightly.api.openapi_generated.swagger_client.model.mongo_object_id import MongoObjectID


class DatasetsApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client
        self.create_dataset_endpoint = _Endpoint(
            settings={
                'response_type': (CreateEntityResponse,),
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets',
                'operation_id': 'create_dataset',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_create_request',
                ],
                'required': [
                    'dataset_create_request',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_create_request':
                        (DatasetCreateRequest,),
                },
                'attribute_map': {
                },
                'location_map': {
                    'dataset_create_request': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client
        )
        self.delete_dataset_by_id_endpoint = _Endpoint(
            settings={
                'response_type': None,
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}',
                'operation_id': 'delete_dataset_by_id',
                'http_method': 'DELETE',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                ],
                'required': [
                    'dataset_id',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                },
                'location_map': {
                    'dataset_id': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )
        self.get_dataset_by_id_endpoint = _Endpoint(
            settings={
                'response_type': (DatasetData,),
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}',
                'operation_id': 'get_dataset_by_id',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                ],
                'required': [
                    'dataset_id',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                },
                'location_map': {
                    'dataset_id': 'path',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )
        self.get_datasets_endpoint = _Endpoint(
            settings={
                'response_type': ([DatasetData],),
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets',
                'operation_id': 'get_datasets',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                ],
                'required': [],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                },
                'attribute_map': {
                },
                'location_map': {
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )
        self.get_datasets_enriched_endpoint = _Endpoint(
            settings={
                'response_type': ([DatasetDataEnriched],),
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/enriched',
                'operation_id': 'get_datasets_enriched',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'limit',
                ],
                'required': [],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'limit':
                        (int,),
                },
                'attribute_map': {
                    'limit': 'limit',
                },
                'location_map': {
                    'limit': 'query',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )
        self.register_dataset_upload_by_id_endpoint = _Endpoint(
            settings={
                'response_type': None,
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}/registerDatasetUpload',
                'operation_id': 'register_dataset_upload_by_id',
                'http_method': 'PUT',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                    'job_status_meta',
                ],
                'required': [
                    'dataset_id',
                    'job_status_meta',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                    'job_status_meta':
                        (JobStatusMeta,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                },
                'location_map': {
                    'dataset_id': 'path',
                    'job_status_meta': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client
        )
        self.update_dataset_by_id_endpoint = _Endpoint(
            settings={
                'response_type': None,
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}',
                'operation_id': 'update_dataset_by_id',
                'http_method': 'PUT',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                    'dataset_update_request',
                ],
                'required': [
                    'dataset_id',
                    'dataset_update_request',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                    'dataset_update_request':
                        (DatasetUpdateRequest,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                },
                'location_map': {
                    'dataset_id': 'path',
                    'dataset_update_request': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client
        )
        self.update_meta_data_configuration_id_by_dataset_id_endpoint = _Endpoint(
            settings={
                'response_type': None,
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}/setMetaDataConfiguration',
                'operation_id': 'update_meta_data_configuration_id_by_dataset_id',
                'http_method': 'PUT',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                    'entity_body',
                ],
                'required': [
                    'dataset_id',
                    'entity_body',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                    'entity_body':
                        (EntityBody,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                },
                'location_map': {
                    'dataset_id': 'path',
                    'entity_body': 'body',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client
        )

    def create_dataset(
        self,
        dataset_create_request,
        **kwargs
    ):
        """create_dataset  # noqa: E501

        Creates a new dataset for a user  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.create_dataset(dataset_create_request, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_create_request (DatasetCreateRequest):

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            CreateEntityResponse
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_create_request'] = \
            dataset_create_request
        return self.create_dataset_endpoint.call_with_http_info(**kwargs)

    def delete_dataset_by_id(
        self,
        dataset_id,
        **kwargs
    ):
        """delete_dataset_by_id  # noqa: E501

        Delete a specific dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.delete_dataset_by_id(dataset_id, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_id (MongoObjectID): ObjectId of the dataset

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            None
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_id'] = \
            dataset_id
        return self.delete_dataset_by_id_endpoint.call_with_http_info(**kwargs)

    def get_dataset_by_id(
        self,
        dataset_id,
        **kwargs
    ):
        """get_dataset_by_id  # noqa: E501

        Get a specific dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_dataset_by_id(dataset_id, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_id (MongoObjectID): ObjectId of the dataset

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            DatasetData
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_id'] = \
            dataset_id
        return self.get_dataset_by_id_endpoint.call_with_http_info(**kwargs)

    def get_datasets(
        self,
        **kwargs
    ):
        """get_datasets  # noqa: E501

        Get all datasets for a user  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_datasets(async_req=True)
        >>> result = thread.get()


        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            [DatasetData]
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        return self.get_datasets_endpoint.call_with_http_info(**kwargs)

    def get_datasets_enriched(
        self,
        **kwargs
    ):
        """get_datasets_enriched  # noqa: E501

        Get all datasets for a user but enriched with additional information as nTags, nEmbeddings, samples  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_datasets_enriched(async_req=True)
        >>> result = thread.get()


        Keyword Args:
            limit (int): if set, only returns the newest up until limit. [optional]
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            [DatasetDataEnriched]
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        return self.get_datasets_enriched_endpoint.call_with_http_info(**kwargs)

    def register_dataset_upload_by_id(
        self,
        dataset_id,
        job_status_meta,
        **kwargs
    ):
        """register_dataset_upload_by_id  # noqa: E501

        Registers a job to track the dataset upload  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.register_dataset_upload_by_id(dataset_id, job_status_meta, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_id (MongoObjectID): ObjectId of the dataset
            job_status_meta (JobStatusMeta):

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            None
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_id'] = \
            dataset_id
        kwargs['job_status_meta'] = \
            job_status_meta
        return self.register_dataset_upload_by_id_endpoint.call_with_http_info(**kwargs)

    def update_dataset_by_id(
        self,
        dataset_id,
        dataset_update_request,
        **kwargs
    ):
        """update_dataset_by_id  # noqa: E501

        Update a specific dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.update_dataset_by_id(dataset_id, dataset_update_request, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_id (MongoObjectID): ObjectId of the dataset
            dataset_update_request (DatasetUpdateRequest): updated data for dataset

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            None
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_id'] = \
            dataset_id
        kwargs['dataset_update_request'] = \
            dataset_update_request
        return self.update_dataset_by_id_endpoint.call_with_http_info(**kwargs)

    def update_meta_data_configuration_id_by_dataset_id(
        self,
        dataset_id,
        entity_body,
        **kwargs
    ):
        """update_meta_data_configuration_id_by_dataset_id  # noqa: E501

        Sets the id of the metadata configuration for a specific datasetId  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.update_meta_data_configuration_id_by_dataset_id(dataset_id, entity_body, async_req=True)
        >>> result = thread.get()

        Args:
            dataset_id (MongoObjectID): ObjectId of the dataset
            entity_body (EntityBody):

        Keyword Args:
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            None
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['dataset_id'] = \
            dataset_id
        kwargs['entity_body'] = \
            entity_body
        return self.update_meta_data_configuration_id_by_dataset_id_endpoint.call_with_http_info(**kwargs)
