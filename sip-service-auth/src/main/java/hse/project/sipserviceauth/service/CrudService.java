package hse.project.sipserviceauth.service;

import hse.project.sipserviceauth.exception.ApiRequestException;
import hse.project.sipserviceauth.model.response.CreateResponse;
import hse.project.sipserviceauth.model.response.DeleteResponse;
import hse.project.sipserviceauth.model.response.UpdateResponse;

import java.util.List;
import java.util.UUID;

public interface CrudService<T, Request> {

    CreateResponse<T> create(Request request) throws ApiRequestException;

    List<T> readAll();

    T readById(UUID id);

    UpdateResponse<T> updateById(UUID id, Request updated) throws ApiRequestException;

    DeleteResponse<T> deleteById(UUID id);
}
