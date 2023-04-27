package hse.project.sipserviceauth.service.orders;

import hse.project.sipserviceauth.SipServiceAuthApplication;
import hse.project.sipserviceauth.exception.ApiRequestException;
import hse.project.sipserviceauth.model.domain.Order;
import hse.project.sipserviceauth.model.request.OrderRequest;
import hse.project.sipserviceauth.model.response.CreateResponse;
import hse.project.sipserviceauth.model.response.DeleteResponse;
import hse.project.sipserviceauth.model.response.UpdateResponse;
import hse.project.sipserviceauth.repository.OrderRepository;

import hse.project.sipserviceauth.service.CrudService;
import hse.project.sipserviceauth.utils.AuthorizedUser;

import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Service;

import java.util.Date;
import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class OrderService implements CrudService<Order, OrderRequest> {

    private final OrderRepository orderRepository;

    @Override
    public CreateResponse<Order> create(OrderRequest orderRequest) throws ApiRequestException {
        Order order = Order.builder()
                .url(orderRequest.getUrl())
                .name(orderRequest.getName())
                .model(orderRequest.getModel())
                .satellite(orderRequest.getSatellite())
                .createdAt(new Date())
                .finishedAt(null)
                .status(false)
                .diff(null)
                .url2(orderRequest.getUrl2())
                .user(AuthorizedUser.getUser())
                .build();

        orderRepository.save(order);
        SipServiceAuthApplication.orders.add(order);

        return new CreateResponse<>("New order created!", order);
    }

    @Override
    public List<Order> readAll() {
        return orderRepository.findAll();
    }

    @Override
    public Order readById(UUID orderId) {
        return orderRepository.findById(orderId).orElse(null);
    }

    @Override
    public UpdateResponse<Order> updateById(UUID id, OrderRequest updated) throws ApiRequestException {
        //TODO: OrderService - implement update order by id method
        return null;
    }

    @Override
    public DeleteResponse<Order> deleteById(UUID orderId) {
        if (orderRepository.existsById(orderId)) {
            orderRepository.deleteById(orderId);
            return new DeleteResponse<>("Order deleted!", orderId);
        }

        return null;
    }
}
