package hse.project.sipserviceauth.api;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.UUID;

public interface CrudController<Request> {

    ResponseEntity<?> create(@RequestBody Request request);

    ResponseEntity<?> readAll();

    ResponseEntity<?> readById(@PathVariable UUID id);

    ResponseEntity<?> updateById(@PathVariable UUID id, @RequestBody Request request);

    ResponseEntity<?> deleteById(@PathVariable UUID id);
}
