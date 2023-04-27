package hse.project.sipserviceauth.model.domain;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import jakarta.persistence.*;

import lombok.*;

import org.hibernate.Hibernate;

import java.util.Date;
import java.util.Objects;
import java.util.UUID;

@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
@Getter
@Setter
@ToString
@RequiredArgsConstructor
@Builder
@AllArgsConstructor
@Entity(name = "order")
@Table(name = "orders")
public class Order {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    private String name;

    @Column(length = 1000000)
    private String url;

    private String model;

    private String satellite;

    private Date createdAt;

    private Date finishedAt;

    private boolean status;

    @Column(length = 1000000)
    private String result;

    @Column(length = 1000000)
    private String bbox;

    @Column(length = 1000000)
    private String url2;

    @Column(length = 1000000)
    private String result2;

    @Column(length = 100000000)
    private String diff;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", referencedColumnName = "user_id", nullable = false)
    @ToString.Exclude
    private User user;

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || Hibernate.getClass(this) != Hibernate.getClass(o)) return false;
        Order order = (Order) o;
        return id != null && Objects.equals(id, order.id);
    }

    @Override
    public int hashCode() {
        return getClass().hashCode();
    }
}
