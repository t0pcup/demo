package hse.project.sipserviceauth.api.v1;

import hse.project.sipserviceauth.api.CrudController;
import hse.project.sipserviceauth.exception.ApiRequestException;
import hse.project.sipserviceauth.model.domain.Order;
import hse.project.sipserviceauth.model.domain.User;
import hse.project.sipserviceauth.model.request.OrderRequest;
import hse.project.sipserviceauth.model.response.CreateResponse;
import hse.project.sipserviceauth.model.response.DeleteResponse;
import hse.project.sipserviceauth.service.orders.OrderService;
import hse.project.sipserviceauth.utils.AuthorizedUser;

import lombok.RequiredArgsConstructor;

import org.aspectj.weaver.ast.Or;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping()
@CrossOrigin(origins = "http://127.0.0.1:5173")
@RequiredArgsConstructor
public class OrderController implements CrudController<OrderRequest> {

    private final OrderService orderService;

    @Override
    @PostMapping("/order")
    @PreAuthorize("hasAnyRole('ROLE_USER', 'ROLE_ADMIN')")
    public ResponseEntity<?> create(@RequestBody OrderRequest orderRequest) {
        CreateResponse<Order> createResponse;

        try {
            createResponse = orderService.create(orderRequest);
        } catch (ApiRequestException e) {
            return new ResponseEntity<>(e.getMessage(), HttpStatus.BAD_REQUEST);
        }

        return new ResponseEntity<>(createResponse, HttpStatus.CREATED);
    }

    @Override
    @GetMapping("/admin/orders")
    @PreAuthorize("hasAnyRole('ROLE_ADMIN')")
    public ResponseEntity<?> readAll() {
        final List<Order> orders = orderService.readAll();

        return orders != null
                ? new ResponseEntity<>(orders, HttpStatus.OK)
                : new ResponseEntity<>("There are NO orders!", HttpStatus.NOT_FOUND);
    }

    @Override
    @GetMapping("/order")
    @PreAuthorize("hasAnyRole('ROLE_ADMIN')")
    public ResponseEntity<?> readById(@RequestParam UUID orderId) {
        final Order order = orderService.readById(orderId);

        return order != null
                ? new ResponseEntity<>(order, HttpStatus.OK)
                : new ResponseEntity<>("No order with such id!", HttpStatus.NOT_FOUND);
    }

    @Override
    public ResponseEntity<?> updateById(UUID id, OrderRequest orderRequest) {
        //TODO: implement update order by id method
        return null;
    }

    @Override
    @DeleteMapping("/order")
    @PreAuthorize("hasAnyRole('ROLE_ADMIN')")
    public ResponseEntity<?> deleteById(UUID orderId) {
        final DeleteResponse<Order> deleteResponse = orderService.deleteById(orderId);

        return deleteResponse != null
                ? new ResponseEntity<>(deleteResponse, HttpStatus.OK)
                : new ResponseEntity<>("No order with such id!", HttpStatus.NOT_FOUND);
    }

    @GetMapping("/orders")
    @PreAuthorize("hasAnyRole('ROLE_USER', 'ROLE_ADMIN')")
    public ResponseEntity<?> getMyOrders() {
        User user = AuthorizedUser.getUser();

        if (user == null) {
            return new ResponseEntity<>("You are NOT authorized!", HttpStatus.UNAUTHORIZED);
        }

        List<Order> readyOrders = new ArrayList<>();
        List<Order> notReadyOrders = new ArrayList<>();
        for (Order order : user.getOrders()) {
            if (order.isStatus()) {
                readyOrders.add(order);
            } else {
                notReadyOrders.add(order);
            }
        }

        readyOrders.sort(Comparator.comparing(Order::getCreatedAt));

        return new ResponseEntity<>(new ArrayList<>(List.of(notReadyOrders, readyOrders)), HttpStatus.OK);
    }

    @DeleteMapping("/orders")
    @PreAuthorize("hasAnyRole('ROLE_USER', 'ROLE_ADMIN')")
    public ResponseEntity<?> deleteOrderById(@RequestParam("order_id") UUID orderId) {
        User user = AuthorizedUser.getUser();

        if (user == null) {
            return new ResponseEntity<>("You are NOT authorized!", HttpStatus.UNAUTHORIZED);
        }

        final DeleteResponse<Order> deleteResponse = orderService.deleteById(orderId);

        return deleteResponse != null
                ? new ResponseEntity<>(deleteResponse, HttpStatus.OK)
                : new ResponseEntity<>("Order with such id not found!", HttpStatus.NOT_FOUND);
    }
}
